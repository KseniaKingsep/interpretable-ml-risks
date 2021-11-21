import math
import pickle
import seaborn as sns
from typing import Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score, precision_score
from sklearn.impute import SimpleImputer

# TODO switch to yaml config

# lgbm_params = {
#                 'learning_rate': [0.01,0.03,0.05,0.1,0.15,0.2],
#                 'max_depth': [6,8,10,12,16],
#                 'num_leaves': [8,16,32,64,96,144],
#                 'feature_fraction': [0.4, 0.8, 1],
#                 'subsample': [0.4, 0.8, 1],
#                 'is_unbalance': [True],
#               }
lgbm_params = {
                'learning_rate': [0.03,0.1],
                'max_depth': [6,10,16],
                'num_leaves': [32],
                'feature_fraction': [0.6],
                'subsample': [0.6],
                'is_unbalance': [True],
              }


class CaseCI:

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
        """

        :return:
        """
        if self.path is None:
            path = '/Users/kseniia/OneDrive/MSDS/Thesis/'
            add = 'data/Case_CI_0/'
            data = pd.read_csv(path + add + 'Case_CI_0_data.csv', sep=',', index_col=0)
        else:
            data = pd.read_csv(self.path, sep=',', index_col=0)
        return data


    def get_names(self):
        self.names = list(self.X_train.columns)

    def train_model(self) -> None:

        self.data = self.load_data()

        self.X_train = self.data.loc[self.data['train_flag'] == 1, :].drop(columns=['target','train_flag'])
        self.y_train = self.data.loc[self.data['train_flag'] == 1, 'target'].drop(columns=['train_flag'])

        self.X_test = self.data.loc[self.data['train_flag'] == 0, :].drop(columns=['target', 'train_flag'])
        self.y_test = self.data.loc[self.data['train_flag'] == 0, 'target'].drop(columns=['train_flag'])

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train,
                                            test_size=0.2, shuffle=True, random_state=17)

        self.get_names()

        self.imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.imp_mean.fit(self.X_train)
        self.X_train = pd.DataFrame(self.imp_mean.transform(self.X_train), columns=self.names)
        self.X_val = pd.DataFrame(self.imp_mean.transform(self.X_val), columns=self.names)
        self.X_test = pd.DataFrame(self.imp_mean.transform(self.X_test), columns=self.names)

        self.grid_pipe_lgbm = GridSearchCV(LGBMClassifier(), lgbm_params, cv=5, scoring='f1', verbose=1, n_jobs=1)
        self.grid_pipe_lgbm.fit(self.X_train, self.y_train)

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
        "ROC AUC train : ", roc_auc_score(self.y_train, pipe.predict(self.X_train)), '\n',
        "ROC AUC val : ", roc_auc_score(self.y_val, pipe.predict(self.X_val)), '\n',
        "ROC AUC test : ", roc_auc_score(self.y_test, pipe.predict(self.X_test)), '\n'
        )

        print("Precision test : ", precision_score(self.y_test, pipe.predict(self.X_test)))
        print("Recall test : ", recall_score(self.y_test, pipe.predict(self.X_test)))

        confusion_matrix = pd.crosstab(self.y_test, pipe.predict(self.X_test),
                                       rownames=['Actual'], colnames=['Predicted'])
        sns.heatmap(confusion_matrix, annot=True)
        plt.show()
        # print(confusion_matrix(self.y_test, pipe.predict(self.X_test)))

    def save_pipe(self, n, name):
        addstr = 'full' if self.full else 'reduced'
        with open(f'../models/pipe_{addstr}_{name}_{n}.pkl', 'wb') as f:
            pickle.dump(self.grid_pipe_lgbm.best_estimator_, f)
