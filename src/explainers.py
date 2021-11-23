from sklearn.metrics import recall_score, precision_score
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
from src.timelimit import *
from tqdm.notebook import tqdm
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

        self.c = 0
        self.model = model
        self.desired_class = desired_class
        self.instances_to_change = [i for i, x in
                   enumerate((model.grid_pipe_lgbm.predict(model.X_test) == desired_class))
                   if x]

    def create_explainer(self):

        X_dice = self.model.X_train.copy()
        X_dice['target'] = self.model.y_train
        d = dice_ml.Data(dataframe=X_dice,
                         continuous_features=self.model.names,
                         outcome_name='target')
        m = dice_ml.Model(model=self.model.grid_pipe_lgbm, backend='sklearn')
        self.exp = dice_ml.Dice(d, m)

    def get_cf(self, instance_id=None, n_cf=5, features_to_vary=None, print=True):

        self.res = None

        if features_to_vary is None:
            features_to_vary = self.model.names

        if instance_id is None:
            instance_id = random.choice(self.instances_to_change)
        query_instance = self.model.X_test.iloc[[instance_id]]

        try:
            with time_limit(5):
                self.res = self.exp.generate_counterfactuals(query_instance, total_CFs=n_cf,
                                                   desired_class="opposite", verbose=False,
                                                   features_to_vary=features_to_vary
                                                   )

                if print and self.res:
                    self.res.visualize_as_dataframe(show_only_changes=True)

        except TimeoutException as e:
            self.c += 1
            pass

    def evaluate_dataset(self, features_to_vary=None, n=None, save=True, name=''):

        self.subset = self.instances_to_change
        if n is not None:
            if n < 0:
                n = int(len(self.model.X_test) * n)
            subset = random.sample(self.instances_to_change, n)

        self.c = 0
        self.cfs = {}
        for row in tqdm(self.subset):
            self.get_cf(instance_id=row, n_cf=5, features_to_vary=features_to_vary, print=False)

            if self.res:
                if self.res.cf_examples_list[0].final_cfs_df is not None:
                    print(f""" Found CF for {len(self.cfs) + 1} rows""")
                    self.cfs[row] = self.res.cf_examples_list[0].final_cfs_df.to_dict('records')

        if save:
            if name == '':
                name = f'test_{datetime.datetime.now()}'
            with open(f'../results/{name}.pkl', 'wb') as f:
                pickle.dump(self.cfs, f, pickle.HIGHEST_PROTOCOL)

        self.print_metrics()

    def print_metrics(self):
        """
        Print out key values
        :return:
        """
        print(f"""{len(self.subset)} instances analyzed""")
        print(f"""{len(self.cfs) / len(self.subset)}% of successfull explanations""")
        print(f"""{self.c / len(self.subset)}% of programming package errors""")
        print(
            f"""{(len(self.subset) - self.c - len(self.cfs)) / len(self.subset)}% of cases, where no CFs could be found by DiCE""")
        self.additional_good_class = self.get_additional_conversion()
        print(f"""{int(self.additional_good_class)} additional good class instances obtained""")
        print(
            f"""{round(self.additional_good_class / len(self.subset), 2)}% of additional successes (model quality adjusted)""")

    def analyze_counterfactuals(self):
        """
        See how the changes in data affect the target
        :return:
        """
        pass


    def get_additional_conversion(self):
        """
        See how many instances of not desired class will be converted to desired
        :return:
        """
        pr = precision_score(self.model.y_val, self.model.grid_pipe_lgbm.predict(self.model.X_val))
        inverse_pr = precision_score(1-self.model.y_val, 1-self.model.grid_pipe_lgbm.predict(self.model.X_val))
        return len(self.cfs) * inverse_pr * pr

    def global_explainers(self):
        """
        Compare the result
        :return:
        """
        pass

    def local_explainers(self):
        """

        :return:
        """
        pass