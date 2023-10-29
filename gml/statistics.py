import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns

from scipy.linalg import inv
from scipy.stats import chi2


class StatisticalAnalysis:
    """
    Class used to extract information from the dataset and better understand it.
    Attributes
    ----------
    name : str
        first name of the person
    surname : str
        family name of the person
    age : int
        age of the person

    Methods
    -------
    info(additional=""):
        Prints the person's name and age.
    """
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset

    def get_counts(self, col_name: str):
        return self.dataset[col_name].value_counts()

    def get_groupby_size_max(self, grouping_column: str):
        return self.dataset.groupby([grouping_column]).size().max()

    def get_groupby_size_min(self, grouping_column: str):
        return self.dataset.groupby([grouping_column]).size().min()

    def get_unique_values(self, col_name: str):
        return self.dataset[col_name].unique()


class StatisticalPlots:
    def __init__(self):
        pass

    @staticmethod
    def plot_bar(bars_coord, heights, x_labels, width=0.8, color=None,
                 plot_x_label='x - axis', plot_y_label='y - axis', plot_title='My bar chart!'):
        if not color:
            color = ['red', 'green']

        plt.bar(bars_coord, heights, tick_label=x_labels, width=width, color=color)
        plt.xlabel(plot_x_label)
        plt.ylabel(plot_y_label)
        plt.title(plot_title)
        plt.show()

    @staticmethod
    def plot_pie(legend, slices, colors):

        plt.pie(slices, labels=legend, colors=colors,
                startangle=90, shadow=True, explode=(0, 0.1),
                radius=1.2, autopct='%1.1f%%')

        plt.legend()
        plt.show()

    @staticmethod
    def plot_cofusion_matrix(conf_matrix):

        plt.figure(figsize=(8, 8))
        sns.set(font_scale=1.5)

        ax = sns.heatmap(
            conf_matrix,  # confusion matrix 2D array
            annot=True,  # show numbers in the cells
            fmt='d',  # show numbers as integers
            cbar=False,  # don't show the color bar
            cmap='flag'  # customize color map
        )

        ax.set_xlabel("Predicted", labelpad=20)
        ax.set_ylabel("Actual", labelpad=20)
        plt.show()


class StatisticalOutlierDetection:

    def __init__(self):
        pass

    def zscore(self, data):
        # apply zscore over amounts after grouping by payment type
        zscores = stats.zscore(data)

        return zscores

    def zscore_outliers(self, zscores, std_threshold=3):
        right_outliers = zscores.index[zscores > std_threshold].tolist()
        left_outliers = zscores.index[zscores < -std_threshold].tolist()

        outliers_indices = left_outliers + right_outliers
        return outliers_indices

    def mahalanobis_distance(self, df):
        """
        Calculate Mahalanobis Distance for a Pandas DataFrame.

        Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data points.

        Returns:
        distances (pandas.Series): A Series containing the Mahalanobis Distance for each data point.
        """
        # Convert the DataFrame to a NumPy array for efficient calculations
        data = df.values

        # Calculate the mean vector
        mean = np.mean(data, axis=0)

        # Calculate the covariance matrix
        covariance_matrix = np.cov(data, rowvar=False)

        # Calculate the inverse of the covariance matrix
        inv_covariance_matrix = inv(covariance_matrix)

        # Initialize an empty list to store the Mahalanobis Distances
        distances = []

        # Calculate the Mahalanobis Distance for each data point
        for i in range(len(data)):
            x_minus_mean = data[i] - mean
            distance = np.sqrt(np.dot(np.dot(x_minus_mean, inv_covariance_matrix), x_minus_mean))
            distances.append(distance)

        # Convert the list of distances to a Pandas Series
        distances = pd.Series(distances, index=df.index, name='MahalanobisDistance')

        return distances

    def mahalanobis_dist_outliers(self, df, mahalanobis_distances):
        """
        Chi-Square Percent Point Function (ppf) helps find a critical value that tells us how extreme or
        unusual a result is in a Chi-Square distribution, often used in statistics.
        It's like a threshold that helps decide if something is very different from what we'd expect by chance.
        :param df: the dataframe used for mahalanobis distances.
        :param mahalanobis_distances: the computed mahalanobis distances.
        :return: outliers from the initial dataset
        """

        # Define the degrees of freedom (df) for the Chi-Square distribution
        df_chi2 = df.shape[1]  # Equal to the number of features

        # Specify the desired significance level (alpha)
        alpha = 0.0001  # You can adjust this value based on your desired level of significance

        # Calculate the Chi-Square critical value for the given significance level
        chi2_critical_value = chi2.ppf(1 - alpha, df_chi2)

        # Detect outliers based on the Mahalanobis Distance and Chi-Square critical value
        outliers_indices = df.index[mahalanobis_distances > np.sqrt(chi2_critical_value)].tolist()
        #df[mahalanobis_distances > np.sqrt(chi2_critical_value)]

        return outliers_indices