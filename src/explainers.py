from sklearn.metrics import recall_score, precision_score
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
from src.timelimit import *
from tqdm import tqdm
import datetime
import dice_ml
import random
import pickle

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

    def explain(self, instance_id=None, n_cf=5, features_to_vary=None, print=True):

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
        except TimeoutException as e:
            print("Timed out!")
            print(f"""Problem for item {row}""")
            pass

        if print:
            self.res.visualize_as_dataframe(show_only_changes=True)

    def evaluate_dataset(self, features_to_vary=None, n=None, save=True, name=''):

        subset = self.instances_to_change
        if n is not None:
            if n < 0:
                n = int(len(self.model.X_test) * n)
            subset = random.sample(self.instances_to_change, n)

        c = 0
        self.cfs = {}
        for row in tqdm(subset):
            self.explain(instance_id=row, n_cf=5, features_to_vary=features_to_vary, print=False)

            if self.res.cf_examples_list[0].final_cfs_df is not None:
                print(f""" Found CF for {len(cfs) + 1} rows""")
                self.cfs[row] = self.res.cf_examples_list[0].final_cfs_df.to_dict('records')

            c += 1

        if save:
            if name == '':
                name = f'test_{datetime.datetime.now()}'
            with open(f'../results/{name}.pkl', 'wb') as f:
                pickle.dump(self.cfs, f, pickle.HIGHEST_PROTOCOL)


        print(f"""% of successfull explanations {len(self.cfs)/len(subset)}""")
        self.additional_good_class = self.get_additional_conversion()
        print(f"""Additional good class via CFs: {self.additional_good_class}, precision {precision_score(self.model.y_val, 
                                                       self.model.grid_pipe_lgbm.predict(self.model.X_val))}""")
        print(f"""% of additional successes {len(self.additional_good_class) / len(subset)}""")


    def analyze_counterfactuals(self):
        """

        :return:
        """
        pass


    def get_additional_conversion(self):
        """
        See how many instances of not desired class will be converted to desired
        :return:
        """
        pr = precision_score(self.model.y_val,
                             self.model.grid_pipe_lgbm.predict(self.model.X_val))
        additional_conversion = 0

        for i in self.cfs.keys():
            if self.model.y_test.values[i] != self.desired_class:
                additional_conversion += 1
        return additional_conversion * pr

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