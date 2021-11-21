import math
import pickle
import seaborn as sns
from typing import Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score, precision_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer

# TODO switch to yaml config
ord_edu = ['illiterate', 'basic.4y', 'basic.6y', 'basic.9y', 'unknown', 'high.school',
           'professional.course', 'university.degree']

lgbm_params = {
                'learning_rate': [0.01,0.03,0.05,0.1,0.15,0.2],
                'max_depth': [6,8,10,12,16],
                'num_leaves': [8,16,32,64,96,144],
                'feature_fraction': [0.4, 0.8, 1],
                'subsample': [0.4, 0.8, 1],
                'is_unbalance': [True],
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


def get_roc_auc_diff(pipe1, pipe2):
    roc_auc_diff = np.round(roc_auc_score(pipe1.y_test, pipe1.grid_pipe_lgbm.predict(pipe1.X_test_enc))
                            - roc_auc_score(pipe2.y_test, pipe2.grid_pipe_lgbm.predict(pipe2.X_test_enc)), 3)

    print(f"""Adding campaign related features to the model results in {roc_auc_diff} increase in ROC AUC score""")


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, input_array, y=None):
        return self

    def transform(self, input_array, y=None):
        return input_array * 1


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
        self.cont_feats = []
        self.cont_feats_excl = []

        # include current campaign data
        if self.full:
            self.categorical_cols = ['job', 'marital', 'education', 'housing', 'loan',
                                     'contact', 'poutcome', 'month', 'day_of_week']
            self.num_cols = ['pdays', 'emp.var.rate', 'cons.price.idx',
                             'cons.conf.idx', 'euribor3m', 'nr.employed']
            self.other_cols = ['default']
            self.log_cols = ['age', 'campaign', 'previous']

        # exclude current campaign data
        else:
            self.current_campaign_cols = ['contact', 'campaign', 'month', 'day_of_week']
            self.categorical_cols = ['job', 'marital', 'education', 'housing', 'loan', 'poutcome']
            self.num_cols = ['pdays', 'emp.var.rate', 'cons.price.idx',
                                  'cons.conf.idx', 'euribor3m', 'nr.employed']
            self.other_cols = ['default']
            self.log_cols = ['age', 'previous']

    def load_data(self) -> pd.DataFrame:
        """

        :return:
        """
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

    def get_names(self):
        if self.full:
            self.names = list(self.column_transformer.named_transformers_['ohe']
                     .named_steps['onehotencoder'].get_feature_names_out(self.categorical_cols)) \
                     + self.num_cols + self.log_cols + self.other_cols
        else:
            self.names = list(self.column_transformer.named_transformers_['ohe']
                    .named_steps['onehotencoder'].get_feature_names_out(self.categorical_cols)) \
                    + self.num_cols + self.log_cols + self.other_cols

    def train_model(self) -> None:

        self.data = self.load_data()
        self.data = self.preprocess(self.data)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data.drop(columns=['y', 'duration']),
                                                            self.data['y'], test_size=0.2, shuffle=True, random_state=17)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train,
                                                                        test_size=0.2, shuffle=True, random_state=17)

        logarithm_transformer = FunctionTransformer(np.log1p, validate=True)
        log_pipe = make_pipeline(logarithm_transformer, MinMaxScaler())
        cat_pipe = make_pipeline(FunctionTransformer(to_str), OneHotEncoder(handle_unknown='ignore'))

        if self.full:
            self.column_transformer = ColumnTransformer([
                ('ohe', cat_pipe, self.categorical_cols),
                ('scale', IdentityTransformer(),self.num_cols), #MinMaxScaler()
                ('log', log_pipe, self.log_cols),
            ],
                remainder='passthrough', verbose=0
            )
        else:
            self.column_transformer = ColumnTransformer([
                ('ohe', cat_pipe, self.categorical_cols),
                ('scale', IdentityTransformer(), self.num_cols), #MinMaxScaler()
                ('log', log_pipe, self.log_cols),
                ('dropper', 'drop', self.current_campaign_cols),
            ],
                remainder='passthrough', verbose=0
            )

        self.column_transformer.fit(self.X_train)
        self.get_names()
        self.X_train_enc = pd.DataFrame(self.column_transformer.transform(self.X_train), columns=self.names)
        self.X_val_enc = pd.DataFrame(self.column_transformer.transform(self.X_val), columns=self.names)
        self.X_test_enc = pd.DataFrame(self.column_transformer.transform(self.X_test),  columns=self.names)

        self.grid_pipe_lgbm = GridSearchCV(LGBMClassifier(), lgbm_params, cv=5, scoring='f1', verbose=1, n_jobs=5)
        self.grid_pipe_lgbm.fit(self.X_train_enc, self.y_train)

    def get_fimp(self, n: int = 20) -> None:

        df_feature_importance = (
            pd.DataFrame({'importance': self.grid_pipe_lgbm.best_estimator_.feature_importances_,
            }, index=self.names).sort_values('importance', ascending=False)
        )

        plt.figure(figsize=(10, 6))
        sns.barplot(df_feature_importance.head(n).importance,
                    df_feature_importance.head(n).index, palette='mako');

    def print_metrics(self) -> None:

        pipe = self.grid_pipe_lgbm
        print(
        " ROC AUC train : ", roc_auc_score(self.y_train, pipe.predict(self.X_train_enc)), '\n',
        "ROC AUC val : ", roc_auc_score(self.y_val, pipe.predict(self.X_val_enc)), '\n',
        "ROC AUC test : ", roc_auc_score(self.y_test, pipe.predict(self.X_test_enc)), '\n'
        )

        print("Precision test : ", precision_score(self.y_test, pipe.predict(self.X_test_enc)))
        print("Recall test : ", recall_score(self.y_test, pipe.predict(self.X_test_enc)))

        confusion_matrix = pd.crosstab(self.y_test, pipe.predict(self.X_test_enc),
                                       rownames=['Actual'], colnames=['Predicted'])
        sns.heatmap(confusion_matrix, annot=True)
        plt.show()

    def save_pipe(self, n):
        addstr = 'full' if self.full else 'reduced'
        with open(f'../models/pipe_{addstr}_{n}.pkl', 'wb') as f:
            pickle.dump(self.grid_pipe_lgbm.best_estimator_, f)
        with open(f'../models/coltr_{addstr}_{n}.pkl', 'wb') as f:
            pickle.dump(self.column_transformer, f)


