import math
import pickle
import seaborn as sns
from typing import Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score, precision_score


ord_edu = ['illiterate', 'basic.4y', 'basic.6y', 'basic.9y', 'unknown', 'high.school',
           'professional.course', 'university.degree']

# include current campaign data
categorical_cols = ['job', 'marital', 'education', 'housing', 'loan',
                   'contact', 'poutcome', 'month','day_of_week']
num_cols = ['pdays', 'emp.var.rate', 'cons.price.idx',
            'cons.conf.idx', 'euribor3m', 'nr.employed']
other_cols = ['default']
log_cols = ['age', 'campaign', 'previous']

# exclude current campaign data
current_campaign_cols = ['contact', 'campaign', 'month','day_of_week']
categorical_cols_excl = ['job', 'marital', 'education', 'housing', 'loan', 'poutcome']
num_cols_excl = ['pdays', 'emp.var.rate', 'cons.price.idx',
                 'cons.conf.idx', 'euribor3m', 'nr.employed']
other_cols_excl = ['default']
log_cols_excl = ['age', 'previous']

lgbm_params = {
                'lgbmclassifier__learning_rate': [0.01,0.03,0.05,0.1,0.15,0.2],
                'lgbmclassifier__max_depth': [6,8,10,12,16],
                'lgbmclassifier__num_leaves': [8,16,32,64,96,144],
                'lgbmclassifier__feature_fraction': [0.4, 0.8, 1],
                'lgbmclassifier__subsample': [0.4, 0.8, 1],
                'lgbmclassifier__is_unbalance': [True],
              }


def to_str(x):
    return pd.DataFrame(x).astype(str)


def cyclic_transformation(column: pd.DataFrame):
    max_value = column.max()
    sin_values = np.sin(2 * math.pi * column / max_value)
    cos_values = np.cos(2 * math.pi * column / max_value)
    return sin_values, cos_values


def compare_auc(modelfull: Pipeline, modelred: Pipeline) -> None:
    roc_auc_diff = np.round(roc_auc_score(modelfull.y_test, modelfull.predict(modelfull.X_test))
                            - roc_auc_score(modelred.y_test, modelred.predict(modelred.X_test)), 3)

    print(f"""Adding campaign related features to the model resulted in {roc_auc_diff} increase in ROC AUC score""")


class BankMarketingModel:

    def __init__(self, path: Union[None, str] = None, full: bool = True) -> None:
        self.path = path
        self.full = full
        self.grid_pipe_lgbm = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_val = None
        self.y_val = None

    def load_data(self) -> pd.DataFrame:
        if self.path is None:
            path = '/Users/kseniia/OneDrive/MSDS/Thesis/'
            add = 'data/bank-marketing/bank-additional/'
            data = pd.read_csv(path + add + 'bank-additional.csv', sep=';')
        else:
            data = pd.read_csv(self.path, sep=';')
        return data

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        data.replace({'no': 0, 'yes': 1}, inplace=True)
        data.replace({999: -1}, inplace=True)
        data.default.replace({'unknown': 1}, inplace=True)
        return data

    def train_bank_model(self) -> None:

        data = self.load_data()
        data = self.preprocess(data)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data.drop(columns=['y', 'duration']),
                                                            data['y'], test_size=0.2, shuffle=True, random_state=17)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train,
                                                                        test_size=0.2, shuffle=True, random_state=17)

        logarithm_transformer = FunctionTransformer(np.log1p, validate=True)
        log_pipe = make_pipeline(logarithm_transformer, MinMaxScaler())
        cat_pipe = make_pipeline(FunctionTransformer(to_str), OneHotEncoder(handle_unknown='ignore'))

        if self.full:
            column_transformer = ColumnTransformer([
                ('ohe', cat_pipe, categorical_cols),
                ('scale', MinMaxScaler(), num_cols),
                ('log', log_pipe, log_cols),
            ],
                remainder='passthrough', verbose=0
            )
        else:
            column_transformer = ColumnTransformer([
                ('ohe', cat_pipe, categorical_cols_excl),
                ('scale', MinMaxScaler(), num_cols_excl),
                ('log', log_pipe, log_cols_excl),
                ('dropper', 'drop', current_campaign_cols),
            ],
                remainder='passthrough', verbose=0
            )

        self.grid_pipe_lgbm = GridSearchCV(make_pipeline(column_transformer, LGBMClassifier()),
                                      lgbm_params, cv=5, scoring='roc_auc', verbose=2, n_jobs=5)
        self.grid_pipe_lgbm.fit(self.X_train, self.y_train)

    def get_fimp(self) -> None:

        pipe = self.grid_pipe_lgbm

        if self.full:
            names = list(pipe.best_estimator_.named_steps['columntransformer'] \
                .transformers_[0][1][1].get_feature_names(
                categorical_cols)) + num_cols + log_cols + other_cols
        else:
            names = list(pipe.best_estimator_.named_steps['columntransformer'] \
                .transformers_[0][1][1].get_feature_names(
                categorical_cols_excl)) + num_cols_excl + log_cols_excl + other_cols_excl

        df_feature_importance = (
            pd.DataFrame({
                'importance': pipe.best_estimator_._final_estimator.feature_importances_,
            }, index=names)
                .sort_values('importance', ascending=False)
        )

        plt.figure(figsize=(10, 6))
        sns.barplot(df_feature_importance.head(20).importance,
                    df_feature_importance.head(20).index, palette='mako');

        plt.savefig('lgbm_importances-01.png')

    def print_metrics(self) -> None:

        pipe = self.grid_pipe_lgbm
        print(
        "ROC AUC train : ", roc_auc_score(self.y_train, pipe.predict(self.X_train)), '\n',
        "ROC AUC val : ", roc_auc_score(self.y_val, pipe.predict(self.X_val)), '\n',
        "ROC AUC test : ", roc_auc_score(self.y_test, pipe.predict(self.X_test)), '\n'
        )

        print("Precision test : ", precision_score(self.y_test, pipe.predict(self.X_test)))
        print("Recall test : ", recall_score(self.y_test, pipe.predict(self.X_test)))

    def save_pipe(self, n):
        if self.full:
            addstr = 'full'
        else:
            addstr = 'reduced'
        with open(f'../models/pipe_{addstr}_{n}.pkl', 'wb') as f:
            pickle.dump(self.grid_pipe_lgbm.best_estimator_, f)


