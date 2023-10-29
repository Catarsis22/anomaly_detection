import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer

from dbscan import DBSCAN
from sklearn.ensemble import IsolationForest

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


class ModelEvaluation:
    def __init__(self):
        pass

    def confusion_matrix(self, y_test, y_pred):
        return confusion_matrix(y_test, y_pred)

    def f1_score(self, y_test, y_pred, average='weighted'):
        return f1_score(y_test, y_pred, average=average)


class FeatureEngineering:

    def __init__(self, df):
        self.df = df

    def time_step_encoder(self, step_col_name='step'):
        """
        Here's the breakdown of the expression:

        2 * np.pi: This part represents the circumference of a full circle, which is 2π times the radius.
        In radians, 2π represents one complete revolution around the unit circle.

        df['step'] / total_hours: This part computes the fraction of the total hours (total_hours) represented by each "step" value.
        This fraction tells you at which point along the circle's circumference a particular "step" value falls.
        So, (2 * np.pi * df['step'] / total_hours) gives you the angles in radians that correspond to each "step" value,
        considering the entire circle as a reference. It helps represent the cyclical nature of time within a unit circle,
        allowing you to effectively encode time for machine learning applications.

        np.cos and np.sin: Applying np.cos and np.sin to these angles calculates the x and y coordinates, respectively,
        of the points on the unit circle. This is similar to how coordinates are calculated for points on the unit circle in trigonometry.
        np.cos gives you the x-coordinate, which represents the horizontal position.
        np.sin gives you the y-coordinate, which represents the vertical position.

        Encoding: By using these coordinates (cosine and sine values), you're effectively encoding each "step" value as
        a point on the unit circle. This encoding allows you to represent the cyclical nature of time while capturing
        both the phase and amplitude of the cyclical pattern.

        """

        # Total hours in 30 days
        total_hours = self.df[step_col_name].max()

        # Apply the sin-cos transformation
        self.df['step_sin'] = np.sin(2 * np.pi * self.df[step_col_name] / total_hours)
        self.df['step_cos'] = np.cos(2 * np.pi * self.df[step_col_name] / total_hours)

        # Drop the original "step" column
        self.df.drop(step_col_name, axis=1, inplace=True)

    def one_hot_encoder(self, col_name):

        ohe = OneHotEncoder(sparse=False)
        one_hot_encoded = ohe.fit_transform(self.df[[col_name]])

        # Convert the one-hot encoded array to a DataFrame with column names
        one_hot_df = pd.DataFrame(one_hot_encoded, columns=ohe.get_feature_names_out([col_name]))

        # Concatenate the one-hot encoded DataFrame with the original DataFrame
        self.df = pd.concat([self.df, one_hot_df], axis=1)

        # Drop the original column
        self.df.drop(col_name, axis=1, inplace=True)

    def kbins_discretizer(self, col_name):

        n_bins = self.__fd_bins_rule(col_name)

        encoder = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')

        # Fit and transform the "Age" column
        discretized_col = encoder.fit_transform(self.df[[col_name]])

        # Convert the discretized array to a DataFrame with a meaningful column name
        discretized_col_df = pd.DataFrame(discretized_col, columns=[col_name+"_bin"])

        # Concatenate the discretized column with the original DataFrame
        self.df = pd.concat([self.df, discretized_col_df], axis=1)

        # Drop the original column
        self.df.drop(col_name, axis=1, inplace=True)

    def __fd_bins_rule(self, col_name):
        # Calculate the Freedman-Diaconis rule-based number of bins
        n = len(self.df)
        iqr = self.df[col_name].quantile(0.75) - self.df[col_name].quantile(0.25)
        bin_width = 2 * iqr / (n ** (1 / 3))  # Freedman-Diaconis rule

        n_bins = int(round((self.df[col_name].max() - self.df[col_name].min()) / bin_width))

        return n_bins


class ClusteringModel():
    def __init__(self, **kwargs):

        self.eps = 0.5
        self.min_samples = 5
        self.labels = []
        self.core_samples_mask = []

        # Update the DBSCAN parameters using kwargs if provided
        self.__dict__.update(kwargs)

    def fit(self, df):
        self.labels, self.core_samples_mask = DBSCAN(df, self.eps, self.min_samples)


class IsolationForestModel():
    def __init__(self, **kwargs):

        self.n_estimatators = 100
        self.contamination = 0.1
        self.n_jobs = 8
        self.random_state = 22

        # Update the DBSCAN parameters using kwargs if provided
        self.__dict__.update(kwargs)

        self.iso_forest = IsolationForest(n_estimators=self.n_estimatators, contamination=self.contamination, n_jobs=self.n_jobs,
                                          random_state=self.random_state)

    def fit(self, df):
        # Fit the model to the data
        self.iso_forest.fit(df)

    def predict(self, df):
        # Predict outliers (-1 for outliers, 1 for inliers)
        y_pred = self.iso_forest.predict(df)

        return y_pred