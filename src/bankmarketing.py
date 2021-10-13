import math
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score, precision_score


weekday_dict = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7}
month_dict = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7,
              'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
ord_edu = ['illiterate', 'basic.4y', 'basic.6y', 'basic.9y', 'unknown', 'high.school',
           'professional.course', 'university.degree']

# include current campaign data
categorical_cols = ['job', 'marital', 'education', 'housing', 'loan', 'contact',  'poutcome']
num_cols = ['pdays', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
other_cols = ['default', 'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos']
log_cols = ['age', 'campaign', 'previous']

# exclude current campaign data
current_campaign_cols = ['contact', 'campaign']
categorical_cols_excl = ['job', 'marital', 'education', 'housing', 'loan', 'poutcome']
num_cols_excl = ['pdays', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
other_cols_excl = ['default']
log_cols_excl = ['age', 'previous']

lgbm_params = {
                'lgbmclassifier__learning_rate': [0.01,0.03,0.05,0.1,0.15,0.2],
                'lgbmclassifier__max_depth': [6,8,10,12,16,20,30],
                'lgbmclassifier__num_leaves': [8,16,32,64,96,144],
                'lgbmclassifier__feature_fraction': [0.2, 0.4, 0.8, 1],
                'lgbmclassifier__subsample': [0.2, 0.4, 0.8, 1],
                'lgbmclassifier__is_unbalance': [True,False],
              }


def to_str(x):
    return pd.DataFrame(x).astype(str)


def cyclic_transformation(column: pd.DataFrame):
    max_value = column.max()
    sin_values = np.sin(2 * math.pi * column / max_value)
    cos_values = np.cos(2 * math.pi * column / max_value)
    return sin_values, cos_values


class BankMarketingModel:

    def __init__(self, path = None, full: bool = True):
        self.path = path
        self.full = full

    def load_data(self, path=None):
        if path is None:
            path = '/Users/kseniia/OneDrive/MSDS/Thesis/'
            add = 'data/bank-marketing/bank-additional/'
            data = pd.read_csv(path + add + 'bank-additional.csv', sep=';')
        else:
            data = pd.read_csv(path, sep=';')
        return data


    def preprocess(self, data: pd.DataFrame):
        data.replace({'no': 0, 'yes': 1}, inplace=True)
        data.replace({999: -1}, inplace=True)
        data.default.replace({'unknown': 1}, inplace=True)
        data['month'] = data.month.replace(month_dict).astype(float)
        data['day_of_week'] = data.day_of_week.replace(weekday_dict).astype(float)
        data['month_sin'], data['month_cos'] = cyclic_transformation(data.month)
        data['weekday_sin'], data['weekday_cos'] = cyclic_transformation(data.day_of_week)
        data.drop(columns=['month', 'day_of_week'], inplace=True)

        return data

    def train_bank_model(self):

        data = self.load_data()

        data = self.preprocess(data)

        X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['y', 'duration']),
                                                            data['y'], test_size=0.2, shuffle=True)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)

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

        grid_pipe_lgbm = GridSearchCV(make_pipeline(column_transformer, LGBMClassifier()),
                                      lgbm_params, cv=5, scoring='roc_auc', verbose=2, n_jobs=5)
        grid_pipe_lgbm.fit(X_train, y_train)

        return grid_pipe_lgbm


    def get_fimp(self, pipe: Pipeline):

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

        return df_feature_importance

    def print_metrics(self, model, X_train, y_train, X_val, y_val, X_test, y_test):
        print(
            "ROC AUC train : ", roc_auc_score(y_train, model.predict(X_train)),
            "ROC AUC val : ", roc_auc_score(y_val, model.predict(X_val)),
            "ROC AUC test : ", roc_auc_score(y_test, model.predict(X_test))
        )

        print("Precision test : ", precision_score(y_test, model.predict(X_test)))
        print("Recall test : ", recall_score(y_test, model.predict(X_test)))