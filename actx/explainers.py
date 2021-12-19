from sklearn.metrics import recall_score, precision_score
from sklearn.inspection import PartialDependenceDisplay
from lime.lime_tabular import LimeTabularExplainer
from slime.lime_tabular import LimeTabularExplainer as sLimeTabularExplainer
import matplotlib.pyplot as plt
from actx.timelimit import *
from tqdm.notebook import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import compress
import seaborn as sns
import pandas as pd
import numpy as np
import datetime
import dice_ml
import random
import pickle
import shap
import logging

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


# TODO add other explainers from notebook
def plot_pdp_ice(model, train_set, features, title_fill=None, original=None, cfs=None, categorical=False):

    if len(features) > 0:
        if len(features) < 4:
            fig, ax = plt.subplots(figsize=(15, 5))
        elif len(features) < 7:
            fig, ax = plt.subplots(figsize=(15, 10))
        elif len(features) < 10:
            fig, ax = plt.subplots(figsize=(15, 15))
        else:
            fig, ax = plt.subplots(figsize=(15, 20-8))

        display = PartialDependenceDisplay.from_estimator(
            model,
            train_set,
            features,
            kind="both",
            subsample=200,
            n_jobs=3,
            grid_resolution=30,
            random_state=0,
            ice_lines_kw={"color": "tab:blue", "alpha": 0.2, "linewidth": 0.5},
            pd_line_kw={"color": "black", "linestyle": "--", "linewidth": 2},
            ax=ax
        )
        if original and cfs:
            add = 0
            for i, arr in enumerate(display.axes_):
                for j, a in enumerate(arr):
                    if a is not None:
                        display.axes_[i][j].axvline(original[i + j + add], color='red', linestyle=':', label='original')
                        display.axes_[i][j].axvline(cfs[i + j + add], color='green', linestyle=':', label='cf')
                        display.axes_[i][j].legend()
                add = len(display.axes_[i]) - 1


        if title_fill:
            display.figure_.suptitle(
                f"""Partial dependence of target value on {title_fill[0]}\n"""
                f"""for the {title_fill[1]} dataset, with {title_fill[2]}"""
            )
        else:
            if not categorical:
                display.figure_.suptitle(
                    f"""Partial dependence of target value (probability of class 1) (red - original, green - counterfactual)""")
            else:
                display.figure_.suptitle(f"""Partial dependence of target value (probability of class 1)""")
        display.figure_.subplots_adjust(wspace=0.2, hspace=0.4)


class DiCeReport:

    def __init__(self, model, desired_class, features_to_vary, custom_range=True):

        """
        :param model: model training class (trained) like CaseCI or BankMarketingModel
        :param desired_class: a good class, that we want to obtain for max number of instances
        """

        self.c = 0
        self.cfs = {}
        self.model = model
        self.desired_class = desired_class

        self.features_to_vary = features_to_vary
        if self.features_to_vary is None:
            self.features_to_vary = self.model.names

        self.instances_to_change = [i for i, x in
                                    enumerate((model.grid_pipe_lgbm.predict(model.X_test) != desired_class))
                                    if x]

        try:
            self.num_cols = self.model.num_cols
        except:
            self.num_cols = self.model.X_train.columns

        self.permitted_range = None
        if custom_range:
            self.permitted_range = {}
            data = self.model.X_test.copy() if custom_range=='test' else self.model.X_train.copy()
            for i, v in data[self.features_to_vary].items():
                if i not in self.model.categorical_cols:
                    self.permitted_range[i] = [v.quantile(0.05), v.quantile(0.95)]
                else:
                    self.permitted_range[i] = list(v.unique())
        print(self.permitted_range)

    def create_explainer(self, method='random', cont_feats=None):

        """
        Initialize explainer
        :return:
        """
        self.method = method
        X_dice = self.model.X_train.copy()
        X_dice['target'] = self.model.y_train

        if cont_feats is None:
            cont_feats = self.model.names
        d = dice_ml.Data(dataframe=X_dice,
                         continuous_features=cont_feats,
                         outcome_name='target')
        m = dice_ml.Model(model=self.model.grid_pipe_lgbm, backend='sklearn')
        self.exp = dice_ml.Dice(d, m, method)

    def get_cf(self, instance_id=None, n_cf=5, timeout=5, printout=False):

        """
        Get counterfactuals for one instance
        :param instance_id: location in the test dataset in model
        :param n_cf: how many CGs to generate
        :param timeout: time to try CFs generation
        :param printout: print the result
        :return:
        """

        self.res = None

        if instance_id is None:
            instance_id = random.choice(self.instances_to_change)
        query_instance = self.model.X_test.iloc[[instance_id]]

        try:
            with time_limit(timeout):
                self.res = self.exp.generate_counterfactuals(query_instance, total_CFs=n_cf,
                                                             desired_class="opposite", verbose=False,
                                                             features_to_vary=self.features_to_vary,
                                                             permitted_range=self.permitted_range,
                                                             random_seed=7)

                if printout and self.res:
                    self.res.visualize_as_dataframe(show_only_changes=True)

        except TimeoutException as e:
            self.c += 1
            pass

    def evaluate_dataset(self, n=None, save=True, name='', timeout=5, printout=False, n_cf=5):

        """

        :param features_to_vary: changeable features
        :param n: number of instances to sample of fraction of the dataset (if < 0)
        :param save: save dictionary with the CFs
        :param name: name for the saving
        :param timeout:  time to try CFs generation
        :param printout: print the result
        :return:
        """
        random.seed(7)
        self.subset = self.instances_to_change
        if n is not None:
            if n <= 1:
                n = int(len(self.instances_to_change) * n)
            self.subset = random.sample(self.instances_to_change, n)

        self.c = 0
        self.cfs = {}
        for row in tqdm(self.subset):
            self.get_cf(instance_id=row, n_cf=n_cf, timeout=timeout, printout=printout)

            if self.res:
                if self.res.cf_examples_list[0].final_cfs_df is not None:
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

        self.combine_all_cf_diffs()

    def print_metrics(self):
        """
        Print out key values
        :return:
        """
        print(
            f"""{round(len(self.instances_to_change) * 100 / len(self.model.X_test), 2)}% of predicted instances of undesired class in the dataset""")
        print(f"""{len(self.subset)} predicted instances of undesired class analyzed""")
        print(f"""{round(len(self.cfs) * 100 / len(self.subset), 2)}% of successfull explanations, {len(self.cfs)} instances""")
        print(f"""{round(self.c * 100 / len(self.subset), 2)}% of programming package errors""")
        print(
            f"""{round((len(self.subset) - self.c - len(self.cfs)) * 100 / len(self.subset), 2)}% of cases, where no CFs could be found by DiCE""")
        self.additional_good_class = self.get_additional_conversion()
        print(
            f"""{round(int(self.additional_good_class) * 100 / len(self.subset), 2)}% of additional potential successes (model quality adjusted)""")
        print(
            f"""{int(self.additional_good_class)} additional potential successes (model quality adjusted)""")
        # print(
        #     f"""{round(self.additional_good_class / len(self.subset) * len(self.instances_to_change),2)} additional successes (model quality adjusted)""")
        print(f"""{int(len(self.instances_to_change)*len(self.cfs) * 100 / len(self.subset)/100)} instances with recommended actions (extrapolated to the whole dataset)""")
        print(
            f"""{int(int(self.additional_good_class) * 100 / len(self.subset)*len(self.instances_to_change)/100)} additional successes in a test set  (model quality adjusted)""")

        print('')
        self.long_dice = pd.melt(self.alldiffs.reset_index(),
                                 id_vars=['instance_id', 'index'],
                                 value_vars=self.alldiffs.columns, #,[:-1],  # self.features_to_vary,
                                 var_name='feature', value_name='dice_diff')

        for n_feats, cnt in (self.long_dice.dropna().groupby(['instance_id', 'index']
                                                             ).size().value_counts() / (
                                     self.long_dice.dropna().groupby(['instance_id', 'index']).ngroups)
        ).iteritems():
            print(f"""In {round(cnt * 100, 2)}% cases {n_feats} features were changed""")

        print('')
        tmp = self.long_dice.dropna().groupby(['instance_id', 'index'])['feature'].unique().astype(str).value_counts()/(
                                     self.long_dice.dropna().groupby(['instance_id', 'index']).ngroups)
        for feats, cnt in tmp.iteritems():
            print(f"""In {100*round(cnt,2)}% cases {feats} features were changed""")

    def preprocess_cf(self, row, res):
        """
        See how the changes in data affect the target
        :return:
        """
        cfdf = res['cfs']
        df = res['original']
        numeric = [col for col in self.model.names if col not in self.model.categorical_cols]
        # diffs = cfdf[numeric].astype(float)[
        #     (cfdf[numeric].astype(float) - df[numeric].astype(float)).ne(0)]  # .drop(columns=['target'])
        tmp_diffs = cfdf[numeric].astype(float) - df[numeric].astype(float)
        diffs = tmp_diffs.astype(float)[tmp_diffs.ne(0)]
        diffs['instance_id'] = row

        # deal with categorical (including text) features
        if len(self.model.categorical_cols) == 0:
            return diffs
        print(f'try to find categorical_cols')
        diffs_cat = (df[self.model.categorical_cols].astype(str) + ' --> ' + cfdf[self.model.categorical_cols].astype(
            str))
        mask = (cfdf[self.model.categorical_cols] != df[self.model.categorical_cols])
        print(mask)
        changed_cat_cols = list(cfdf[self.model.categorical_cols].columns[mask.values[0]])
        if len(changed_cat_cols) == 0:
            return diffs
        print(f""" !!!!!!!! {changed_cat_cols}""")
        return pd.concat([diffs, diffs_cat[mask]], axis=1)

    def combine_all_cf_diffs(self):

        dfs = []
        for row, res in self.cfs.items():
            dfs.append(self.preprocess_cf(row, res))
        if len(dfs) < 0:
            raise ValueError('No CFs generated')
        self.alldiffs = pd.concat(dfs, axis=0)
        self.alldiffs = self.alldiffs.dropna(axis=1, how='all')

    def plot_diffs(self, x=None, y=None, columns=None, size=3):

        if x is not None and y is not None:
            plt.figure(figsize=(x, y))
            plt.xticks(rotation=45)
        if columns is not None:
            sns.stripplot(data=self.alldiffs.drop(columns='instance_id')[columns],
                          jitter=True, alpha=0.7, marker="h", size=size)
        else:
            sns.stripplot(data=self.alldiffs.drop(columns='instance_id'),
                          jitter=True, alpha=0.7, marker="h", size=size)
        plt.xticks(rotation=45)
        plt.axhline(0)
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

    def __init__(self, model, desired_class, slime=True, features_to_vary=None, use_sep_model=False):

        self.model = model
        self.slime = slime
        self.desired_class = desired_class

        if use_sep_model:
            self.estimator = self.model.grid_pipe_lgbm_sep
        else:
            self.estimator = self.model.grid_pipe_lgbm.best_estimator_

        self.names = self.model.names_sep if use_sep_model else self.model.names
        self.features_to_vary = features_to_vary
        if self.features_to_vary is None:
            self.features_to_vary = self.names

        self.X_test_lime = self.model.X_test_sep if use_sep_model else self.model.X_test
        self.X_train_lime = self.model.X_test_sep if use_sep_model else self.model.X_test

        self.instances_to_change = [i for i, x in enumerate((self.estimator.predict(self.X_test_lime) == desired_class))
                                    if x]

    def create_explainer(self):
        """
        Just initialize LimeTabularExplainer
        :return:
        """
        if self.slime:
            self.exp = sLimeTabularExplainer(self.X_train_lime.values,
                                             mode='classification',
                                             feature_names=self.X_train_lime.columns,
                                             discretize_continuous=False,
                                             feature_selection='lasso_path',
                                             sample_around_instance=True)
        else:
            self.exp = LimeTabularExplainer(self.X_train_lime.values,
                                            mode='classification',
                                            feature_names=self.X_train_lime.columns,
                                            discretize_continuous=False)

    def get_exp(self, instance_id=None, printout=False):
        """
        Explain one particular instance
        :param printout:
        :return:
        """
        if instance_id is None:
            instance_id = random.choice(self.instances_to_change)
        if self.slime:
            self.one_exp = self.exp.slime(self.X_test_lime.iloc[instance_id].values,
                                          self.estimator.predict_proba,
                                          num_features=len(self.X_train_lime.columns), num_samples=1000,
                                          n_max=10000, alpha=0.05)
        else:
            self.one_exp = self.exp.explain_instance(self.X_test_lime.iloc[instance_id],
                                                     self.estimator.predict_proba,
                                                     num_features=len(self.X_train_lime.columns))

        self.lime_pred = pd.DataFrame([t for t in self.one_exp.as_list()
                                       if any(f == t[0] for f in self.features_to_vary)],
                                      columns=['feature', 'lime_coef'])
        self.lime_pred['instance_id'] = instance_id
        if self.desired_class == 0:
            self.lime_pred['lime_coef'] = -self.lime_pred['lime_coef']

        if printout:
            self.one_exp.show_in_notebook(show_table=True, show_all=False)

    def evaluate_dataset(self, cfs: dict = None, n=None, n_exps=5, save=False, name=''):
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
            if self.slime:
                self.get_exp(instance_id=row)
                self.limeexps.append(self.lime_pred)
            else:
                for i in range(n_exps):
                    self.get_exp(instance_id=row)
                    self.limeexps.append(self.lime_pred)
        if self.slime:
            self.allcoefs = pd.concat(self.limeexps, axis=0).sort_values(by=['instance_id', 'feature'])
            self.aggbyidfeat = self.allcoefs.copy()
        else:
            self.allcoefs = pd.concat(self.limeexps, axis=0).sort_values(by=['instance_id', 'feature'])
            self.aggbyidfeat = self.allcoefs.groupby(['instance_id', 'feature']).agg([np.std, np.mean])
            self.aggbyidfeat.columns = self.aggbyidfeat.columns.droplevel(0)
            self.aggbyidfeat = self.aggbyidfeat.rename_axis(None, axis=1).reset_index()

        if save:
            if name == '':
                name = f'test_{datetime.datetime.now()}'
            with open(f'../results/lime_{name}.pkl', 'wb') as f:
                pickle.dump(self.allcoefs, f, pickle.HIGHEST_PROTOCOL)

    def plot_coefs(self, ymax=None, ymin=None):

        plt.figure(figsize=(14, 8))
        if ymax is not None and ymin is not None:
            plt.ylim(ymax, ymin)
        sns.stripplot(data=self.allcoefs, x='feature', y='lime_coef',
                          jitter=0.3, alpha=0.5, marker="h", size=7)

        plt.axhline(0)
        plt.title('Lime coefficients for changeable features')
        plt.legend([], [], frameon=False)
        plt.xticks(rotation=90)


class ShapReport:

    def __init__(self, model, desired_class, features_to_vary=None, use_sep_model=False):
        self.model = model
        self.desired_class = desired_class

        self.names = self.model.names_sep if use_sep_model else self.model.names
        self.features_to_vary = features_to_vary
        if self.features_to_vary is None:
            self.features_to_vary = self.names

        if use_sep_model:
            self.estimator = self.model.grid_pipe_lgbm_sep
        else:
            self.estimator = self.model.grid_pipe_lgbm.best_estimator_

        self.X_test_shap = self.model.X_test_sep if use_sep_model else self.model.X_test

        self.instances_to_change = [i for i, x in enumerate((self.estimator.predict(self.X_test_shap) == desired_class))
                                    if x]

    def create_explainer(self):
        shap.initjs()
        self.explainer = shap.TreeExplainer(self.estimator)
        self.shap_values = self.explainer.shap_values(self.X_test_shap)

    def evaluate_dataset(self, cfs=None, save=False, name=''):

        self.shap_values_ = pd.DataFrame(self.shap_values[self.desired_class], columns=self.names)[
            self.features_to_vary]
        self.shap_values_['instance_id'] = self.shap_values_.index

        if cfs is not None:
            self.subset = list(cfs.keys())
            self.shap_values_ = self.shap_values_[self.shap_values_.instance_id.isin(self.subset)]

        self.shap_long = pd.melt(self.shap_values_, id_vars=['instance_id'],
                                 value_vars=self.features_to_vary,
                                 var_name='feature', value_name='shap_value')

        if save:
            if name == '':
                name = f'test_{datetime.datetime.now()}'
            with open(f'../results/shap_{name}.pkl', 'wb') as f:
                pickle.dump(self.shap_values_, f, pickle.HIGHEST_PROTOCOL)

    def plot_coefs(self, ymax=None, ymin=None):

        plt.figure(figsize=(14, 10))
        if ymax is not None and ymin is not None:
            plt.ylim(ymax, ymin)

        sns.stripplot(x="feature", y="shap_value", data=self.shap_long, alpha=0.5, jitter=0.3, size=7)
        plt.axhline(0)
        plt.title('SHAP coefficients for changeable features')
        plt.legend([], [], frameon=False)
        plt.xticks(rotation=90)


class Comparator:

    def __init__(self, model, desired_class=0, dice=None, sh=None, use_sep_model=False):
        self.dice = dice
        self.sh = sh
        self.features_to_vary = [col for col in dice.alldiffs.columns if col != 'instance_id']
        self.features_to_vary_ohe = self.features_to_vary
        if use_sep_model:
            self.features_to_vary_ohe = [col for col in self.sh.features_to_vary if
                                     any(s in col for s in self.features_to_vary)]
        self.lime = LimeReport(model, desired_class, slime=True,
                               features_to_vary=self.features_to_vary_ohe,
                               use_sep_model=use_sep_model)

        self.model = model

    def compare_dice_slime(self):
        self.lime.create_explainer()
        self.lime.evaluate_dataset(cfs=self.dice.cfs)

        tmp = self.lime.aggbyidfeat.reset_index()
        tmp = tmp.pivot_table(index='instance_id', columns='feature', values='lime_coef').reset_index()

        self.wide = self.dice.alldiffs.merge(tmp, how='left', on='instance_id',
                                             suffixes=['_dice', '_lime'])

        self.long = self.dice.long_dice.merge(self.lime.aggbyidfeat,
                                              how='left', on=['instance_id', 'feature']) \
            .dropna(how='any', axis=0).sort_values(['instance_id', 'index'])
        self.long['one_sign'] = ~((self.long['dice_diff'] > 0) ^ (self.long['lime_coef'] > 0))

        self.sign_correspondence = self.long.groupby('feature')[['one_sign']].mean().sort_values('one_sign')
        self.sign_correspondence.columns = ['% of equal signs']

        print('Share of cases where  SLIME coefficient sign is equal to DiCE suggestion sign:')
        print(self.sign_correspondence.round(4) * 100)

        tmp = self.long.groupby(['index', 'instance_id'])['feature'].apply(lambda x: ', '.join(x)).reset_index()
        print('')
        print(f"""Most common changed by DiCE feature sets are: """)
        for set, cnt in tmp.feature.value_counts().head().iteritems():
            print(f"""Set [{set}] changed in {round(cnt*100 / len(tmp), 2)}% cases""")

    def compare_dice_shap(self):

        self.sh.create_explainer()
        self.sh.evaluate_dataset(cfs=self.dice.cfs)

        self.long_sh = self.dice.long_dice.merge(self.sh.shap_long,
                                                 how='left', on=['instance_id', 'feature']) \
            .dropna(how='any', axis=0).sort_values(['instance_id', 'index'])
        self.long_sh['one_sign'] = ~((self.long_sh['dice_diff'] > 0) ^ (self.long_sh['shap_value'] > 0))

        self.sign_correspondence_sh = self.long_sh.groupby('feature')[['one_sign']].mean().sort_values('one_sign')
        self.sign_correspondence_sh.columns = ['% of equal signs']

        print('Share of cases where SHAP coefficient sign is equal to DiCE suggestion sign:')
        print(self.sign_correspondence_sh.round(4) * 100)

    def compare_lime_shap(self):
        self.lime_shap_long = self.long.merge(self.long_sh, on=['instance_id', 'index', 'feature']
                                              ).query('index==0')
        self.lime_shap_long['dice_diff_x'] = self.lime_shap_long['dice_diff_x'].astype(float)
        print(self.lime_shap_long[['lime_coef', 'shap_value', 'dice_diff_x']].corr())
        print('')

        self.lime_shap_long['one_sign'] = ~((self.lime_shap_long['lime_coef'] > 0) ^
                                            (self.lime_shap_long['shap_value'] > 0))
        self.sign_correspondence_l_sh = self.lime_shap_long.groupby('feature')[['one_sign']]\
                                                            .mean().sort_values('one_sign')
        self.sign_correspondence_l_sh.columns = ['% of equal signs']
        print(self.sign_correspondence_l_sh.round(4) * 100)

    def custom_sorter(self, column):
        correspondence = {team: order for order, team in enumerate(self.features_to_vary)}
        return column.map(correspondence)

    def compare_instance(self, instance_id=None, cf_number=0):
        if instance_id is None:
            instance_id = random.choice(list(self.dice.cfs.keys()))

        # DiCE
        original = self.dice.cfs[instance_id]['original'][self.features_to_vary].values
        cfs = self.dice.cfs[instance_id]['cfs'][self.features_to_vary].iloc[cf_number].values
        mask = (original != cfs)
        changed_features = list(compress(self.features_to_vary, mask))
        changed_features_ohe = self.colnames_to_ohe(changed_features)
        changed_num_features = [f for f in changed_features if f not in self.model.categorical_cols]
        original = list(compress(original, mask))
        cfs = list(compress(cfs, mask))

        if len(changed_features) > 0:
            fig = None
            if len(changed_num_features) > 0:
                fig = make_subplots(rows=1, cols=len(changed_num_features), subplot_titles=changed_num_features)
            for c in range(len(changed_features)):
                if changed_features[c] in self.model.categorical_cols:
                    print(f"""{changed_features[c]}: {original[c]} ---> {cfs[c]}""")
                else:
                    y_rc = [original[c], cfs[c] - original[c], cfs[c]]
                    fig.add_trace(
                        go.Waterfall(x=['original', 'difference', 'counterfactual'], y=y_rc,
                                     orientation="v", name=changed_features[c],
                                     measure=["relative", "relative", "total"],
                                     text=[round(val, 3) for val in y_rc],
                                     connector={"line": {"color": "rgb(63, 63, 63)"}}),
                        row=1, col=c + 1)

            if fig:
                fig.update_layout(height=300, width=800, title_text=f"""DiCE results for instance {instance_id}""",
                                  showlegend=False)
                fig.show()

        # LIME
        self.lime.get_exp(instance_id=instance_id, printout=False)
        tmp = self.lime.lime_pred.sort_values(by='feature', key=self.custom_sorter)
        tmp = tmp[tmp['feature'].isin(changed_features_ohe)]

        fig = go.Figure(go.Bar(
            x=tmp.feature, text=list(tmp.lime_coef.round(4).astype(str).values),
            y=list(tmp.lime_coef.round(4).values),
        ))

        fig.update_layout(height=200, width=800, title=f"""SLIME results for instance {instance_id} """,
                          showlegend=False, margin=dict(l=20, r=20, t=30, b=5))
        fig.show()

        for feat in changed_num_features:
            feat_idx = changed_features.index(feat)
            dice_sign = (cfs[feat_idx] - original[feat_idx]) > 0
            lime_sign = list(tmp.lime_coef.round(4).values)[feat_idx] > 0
            text = 'corresponds' if (dice_sign == lime_sign) else 'doesn\'t correspond'
            mark = u'\u2713' if (dice_sign == lime_sign) else 'x'

            print(f"""[{mark} {feat}] S-LIME sign {text} with DiCE""")

        # SHAP
        tmp_sh = self.sh.shap_long.query('instance_id == @instance_id').sort_values(
                            by='feature', key=self.custom_sorter)
        tmp_sh = tmp_sh[tmp_sh.feature.isin(changed_features_ohe)]

        fig = go.Figure(go.Bar(x=tmp_sh['feature'], y=list(tmp_sh['shap_value'].round(4).values),
                               text=list(tmp_sh['shap_value'].round(4).astype(str).values) ))

        fig.update_layout(height=350, width=800, title=f"""SHAP results for instance {instance_id} """,
                          showlegend=False, margin=dict(l=30, r=20, t=30, b=5))
        fig.show()

        for feat in changed_num_features:
            feat_idx = changed_features.index(feat)
            dice_sign = (cfs[feat_idx] - original[feat_idx]) > 0
            shap_sign = list(tmp_sh.shap_value.round(4).values)[feat_idx] > 0
            text = 'corresponds' if (dice_sign == shap_sign) else 'doesn\'t correspond'
            mark = u'\u2713' if (dice_sign == shap_sign) else 'x'

            print(f"""[{mark} {feat}] SHAP sign {text} with DiCE""")

        # PDP_ICE
        print('')
        plot_pdp_ice(self.model.grid_pipe_lgbm, self.model.X_train, changed_num_features,
                     original=original, cfs=cfs)

        # PDP
        if len(changed_num_features) == 2:
            features = [changed_num_features[0], changed_num_features[1], tuple(changed_num_features)]
            print('')
            print("Computing partial dependence plots...")
            fig, ax = plt.subplots(ncols=3, figsize=(10, 4))

            display = PartialDependenceDisplay.from_estimator(
                self.lime.estimator, self.lime.X_train_lime,
                features, kind="average", n_jobs=3,
                grid_resolution=20, ax=ax
            )

            ax[0].axvline(original[0], color='red', linestyle='--', label='original')
            ax[0].axvline(cfs[0], color='green', linestyle='--', label='counerfactual')
            ax[1].axvline(original[1], color='red', linestyle='--', label='original')
            ax[1].axvline(cfs[1], color='green', linestyle='--', label='counerfactual')

            display.figure_.suptitle(
                "Partial dependence of target value (probability of class 1) \n (red - original, "
                "green - counterfactual) "
            )
            display.figure_.subplots_adjust(wspace=0.4, hspace=0.3)

        if len(self.model.categorical_cols) > 0:
            original = self.dice.cfs[instance_id]['original'][self.features_to_vary].values
            cfs = self.dice.cfs[instance_id]['cfs'][self.features_to_vary].iloc[cf_number].values
            mask = (original != cfs)
            changed_features = list(compress(self.features_to_vary, mask))
            changed_cat_features = [f for f in changed_features if f in self.model.categorical_cols]
            changed_cat_features_ohe = self.colnames_to_ohe(changed_cat_features)

            if len(changed_cat_features_ohe) > 0:
                plot_pdp_ice(self.model.grid_pipe_lgbm_sep, self.model.X_train_sep, changed_cat_features_ohe,
                             categorical=True)

    def colnames_to_ohe(self, colnames):
        colnames_ohe = []
        for colname in colnames:
            for ohe_name in self.model.inner_names:
                if colname in ohe_name:
                    colnames_ohe.append(ohe_name)

        return colnames_ohe