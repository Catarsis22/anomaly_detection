import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns


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
            cmap='flag',  # customize color map
            vmax=175  # to get better color contrast
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
        outliers_indices = np.where(zscores > std_threshold)
        return outliers_indices
