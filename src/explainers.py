from sklearn.metrics import recall_score, precision_score
from sklearn.inspection import PartialDependenceDisplay
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from src.timelimit import *
from tqdm.notebook import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
import datetime
import dice_ml
import random
import pickle

import logging

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


# TODO add other explainers from notebook
def plot_pdp_ice(model, X_train, features, title_fill):
    fig, ax = plt.subplots(figsize=(10, 10))

    display = PartialDependenceDisplay.from_estimator(
        model,
        X_train,
        features,
        kind="both",
        subsample=200,
        n_jobs=3,
        grid_resolution=30,
        random_state=0,
        ice_lines_kw={"color": "tab:blue", "alpha": 0.2, "linewidth": 0.5},
        pd_line_kw={"color": "tab:orange", "linestyle": "--"},
        ax=ax
    )
    display.figure_.suptitle(
        f"""Partial dependence of target value on {title_fill[0]}\n"""
        f"""for the {title_fill[1]} dataset, with {title_fill[2]}"""
    )
    display.figure_.subplots_adjust(wspace=0.2, hspace=0.4)


class DiCeReport:

    def __init__(self, model, desired_class):

        """
        :param model: model training class (trained) like CaseCI or BankMarketingModel
        :param desired_class: a good class, that we want to obtain for max number of instances
        """

        self.c = 0
        self.cfs = {}
        self.model = model
        self.desired_class = desired_class
        self.instances_to_change = [i for i, x in
                                    enumerate((model.grid_pipe_lgbm.predict(model.X_test) == desired_class))
                                    if x]

    def create_explainer(self, method='random'):

        """
        Initialize explainer
        :return:
        """

        X_dice = self.model.X_train.copy()
        X_dice['target'] = self.model.y_train
        d = dice_ml.Data(dataframe=X_dice,
                         continuous_features=self.model.names,
                         outcome_name='target')
        m = dice_ml.Model(model=self.model.grid_pipe_lgbm, backend='sklearn')
        self.exp = dice_ml.Dice(d, m, method)

    def get_cf(self, instance_id=None, n_cf=5, features_to_vary=None,
               timeout=5, printout=False):

        """
        Get counterfactuals for one instance
        :param instance_id: location in the test dataset in model
        :param n_cf: how many CGs to generate
        :param features_to_vary: changeable features
        :param timeout: time to try CFs generation
        :param printout: print the result
        :return:
        """

        self.res = None

        if features_to_vary is None:
            features_to_vary = self.model.names

        if instance_id is None:
            instance_id = random.choice(self.instances_to_change)
        query_instance = self.model.X_test.iloc[[instance_id]]

        try:
            with time_limit(timeout):
                self.res = self.exp.generate_counterfactuals(query_instance, total_CFs=n_cf,
                                                             desired_class="opposite", verbose=False,
                                                             features_to_vary=features_to_vary
                                                             )

                if printout and self.res:
                    self.res.visualize_as_dataframe(show_only_changes=True)

        except TimeoutException as e:
            self.c += 1
            pass

    def evaluate_dataset(self, features_to_vary=None, n=None, save=True, name='',
                         timeout=5, printout=False):

        """

        :param features_to_vary: changeable features
        :param n: number of instances to sample of fraction of the dataset (if < 0)
        :param save: save dictionary with the CFs
        :param name: name for the saving
        :param timeout:  time to try CFs generation
        :param printout: print the result
        :return:
        """

        self.subset = self.instances_to_change
        if n is not None:
            if n < 0:
                n = int(len(self.instances_to_change) * n)
            self.subset = random.sample(self.instances_to_change, n)

        self.c = 0
        self.cfs = {}
        for row in tqdm(self.subset):
            self.get_cf(instance_id=row, n_cf=5, features_to_vary=features_to_vary,
                        timeout=timeout, printout=printout)

            if self.res:
                if self.res.cf_examples_list[0].final_cfs_df is not None:
                    # self.cfs[row] = self.res.cf_examples_list[0].final_cfs_df.to_dict('records')
                    self.cfs[row] = {
                        'original': self.res.cf_examples_list[0].test_instance_df.iloc[0],
                        'cfs': self.res.cf_examples_list[0].final_cfs_df
                    }
                    print(f""" Found CF for {len(self.cfs) + 1} rows""")
        if save:
            if name == '':
                name = f'test_{datetime.datetime.now()}'
            with open(f'../results/dice{name}.pkl', 'wb') as f:
                pickle.dump(self.cfs, f, pickle.HIGHEST_PROTOCOL)

    def print_metrics(self):
        """
        Print out key values
        :return:
        """
        print(
            f"""{len(self.instances_to_change) / len(self.model.X_test)}% of instances of undesired class in the dataset""")
        print(f"""{len(self.subset)} instances of undesired class analyzed""")
        print(f"""{len(self.cfs) * 100 / len(self.subset)}% of successfull explanations""")
        print(f"""{self.c * 100 / len(self.subset)}% of programming package errors""")
        print(
            f"""{(len(self.subset) - self.c - len(self.cfs)) * 100 / len(self.subset)}% of cases, where no CFs could be found by DiCE""")
        self.additional_good_class = self.get_additional_conversion()
        print(
            f"""{round(int(self.additional_good_class) * 100 / len(self.subset), 2)}% of additional successes (model quality adjusted)""")

    def preprocess_cf(self, row, res):
        """
        See how the changes in data affect the target
        :return:
        """
        cfdf = res['cfs']
        df = res['original']
        diffs = cfdf[(cfdf - df).ne(0)].drop(columns='target')
        diffs['instance_id'] = row
        return diffs

    def combine_all_cf_diffs(self):

        dfs = []
        for row, res in self.cfs.items():
            dfs.append(self.preprocess_cf(row, res))
        self.alldiffs = pd.concat(dfs, axis=0)
        self.alldiffs = self.alldiffs.dropna(axis=1, how='all')

    def plot_diffs(self):

        self.combine_all_cf_diffs()

        sns.stripplot(data=self.alldiffs.drop(columns='instance_id'),
                      jitter=True, alpha=0.7, marker="h", size=10)
        plt.title('Changes to variables to obtain counterfactuals')

    def get_additional_conversion(self):
        """
        See how many instances of not desired class will be converted to desired
        :return:
        """
        pr = precision_score(self.model.y_val, self.model.grid_pipe_lgbm.predict(self.model.X_val))
        inverse_pr = precision_score(1 - self.model.y_val, 1 - self.model.grid_pipe_lgbm.predict(self.model.X_val))
        return len(self.cfs) * inverse_pr * pr


class LimeReport:

    def __init__(self, model, desired_class):

        self.model = model
        self.desired_class = desired_class
        self.instances_to_change = [i for i, x in
                                    enumerate((model.grid_pipe_lgbm.predict(model.X_test) == desired_class))
                                    if x]

    def create_explainer(self):
        """
        Just initialize LimeTabularExplainer
        :return:
        """
        self.exp = LimeTabularExplainer(self.model.X_train.values, mode='classification',
                                        feature_names=self.model.X_train.columns,
                                        discretize_continuous=False)

    def get_exp(self, instance_id=None, features_to_vary=None, printout=False):
        """
        Explain one particular instance
        :param features_to_vary:
        :param printout:
        :return:
        """
        if instance_id is None:
            instance_id = random.choice(self.instances_to_change)
        self.one_exp = self.exp.explain_instance(self.model.X_test.iloc[instance_id],
                                                 self.model.grid_pipe_lgbm.predict_proba,
                                                 num_features=len(self.model.X_train.columns))

        if features_to_vary is None:
            features_to_vary = self.model.names
        self.lime_pred = pd.DataFrame([t for t in self.one_exp.as_list()
                                       if any(f == t[0] for f in features_to_vary)],
                                      columns=['feature', 'coef'])
        self.lime_pred['instance_id'] = instance_id
        if self.desired_class == 0:
            self.lime_pred['coef'] = -self.lime_pred['coef']

        if printout:
            self.one_exp.show_in_notebook(show_table=True, show_all=False)

    def evaluate_dataset(self, cfs: dict = None, n=None, n_exps=5, features_to_vary=None, save=False, name=''):
        """
        Generate lime explanations for several instances of a dataset
        :param cfs: dict of counterfactuals if present
        :return:
        """
        if cfs is not None:
            self.subset = list(cfs.keys())
        else:
            self.subset = self.instances_to_change
            if n is not None:
                if n < 0:
                    n = int(len(self.instances_to_change) * n)
                self.subset = random.sample(self.instances_to_change, n)

        self.limeexps = []
        for row in tqdm(self.subset):
            for i in range(n_exps):
                self.get_exp(instance_id=row, features_to_vary=features_to_vary)
                self.limeexps.append(self.lime_pred)
        self.allcoefs = pd.concat(self.limeexps, axis=0).sort_values(by=['instance_id', 'feature'])
        self.aggbyidfeat = self.allcoefs.groupby(['instance_id', 'feature']).agg([np.std, np.mean])
        self.aggbyidfeat.columns = self.aggbyidfeat.columns.droplevel(0)
        self.aggbyidfeat = self.aggbyidfeat.rename_axis(None, axis=1).reset_index()

        if save:
            if name == '':
                name = f'test_{datetime.datetime.now()}'
            with open(f'../results/lime_{name}.pkl', 'wb') as f:
                pickle.dump(self.allcoefs, f, pickle.HIGHEST_PROTOCOL)

    def plot_coefs(self):

        plt.figure(figsize=(10, 6))
        if len(self.subset) <= 30:
            sns.stripplot(data=self.allcoefs, x='feature', y='coef', dodge=True,
                          jitter=True, alpha=0.5, marker="h", size=5, hue='instance_id')
            sns.pointplot(x="feature", y="coef", hue="instance_id",
                          data=self.allcoefs, dodge=.8 - .8 / 3, join=False,
                          markers="d", scale=0.75, ci=None, alpha=0.5)
        else:
            sns.pointplot(x="feature", y="coef", hue="instance_id",
                          data=self.allcoefs, dodge=.8 - .8 / 3, join=False,
                          scale=0.5, ci=None, alpha=0.5)
        sns.pointplot(x="feature", y="coef", color='black',
                      data=self.allcoefs, join=False,
                      markers="d", scale=1, ci='sd', zorder=1000)

        plt.axhline(0)
        plt.title('Lime coefficients for changeable features (colored by instances)')
        plt.legend([], [], frameon=False)


class Comparator:

    def __init__(self, model, desired_class=0, dice=None, shap=None):

        self.dice = dice
        self.lime = LimeReport(model, desired_class)
        self.shap = shap

    def compare_dice_lime(self, features_to_vary):
        self.lime.create_explainer()
        self.lime.evaluate_dataset(cfs=self.dice.cfs, features_to_vary=features_to_vary)
        self.df = self.dice.alldiffs.merge(self.lime.aggbyidfeat, how='left', on='instance_id')

