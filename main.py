import pandas as pd


if __name__ == '__main__':
    dataset = pd.read_csv("./data/Fraud.csv")

    sa = StatisticalAnalysis(dataset)
    sp = StatisticalPlots()
    so = StatisticalOutlierDetection()

    # get NonFraud-Fraud counts
    nonfraud_frad_counts = sa.get_counts("isFraud").values
    print(nonfraud_frad_counts)
    sp.plot_pie(["Not Fraud", "Fraud"], nonfraud_frad_counts.values, ['g', 'r'])

    # get the smallest account based on activity
    print(sa.get_groupby_size_min("nameOrig"))

    # get the biggest account based on activity
    print(sa.get_groupby_size_max("nameOrig"))

    # analyze types of transactions
    print(sa.get_unique_values("type"))
    type_counts = sa.get_counts("type")
    print(type_counts)
    sp.plot_bar(list(range(len(type_counts))), type_counts.values, type_counts.index, color=['r', 'g', 'y', 'b', 'c'])


    # ploting distributions of amount for each payment type
    dataset.loc[dataset.type == "PAYMENT"]["amount"].plot(kind='kde')
    dataset.loc[dataset.type == "PAYMENT"]["amount"].plot(kind='hist', edgecolor='black')

    col_zscores = so.zscore(dataset.loc[dataset.type == "PAYMENT"]["amount"])
    outliers_idx = so.zscore_outliers(col_zscores)
    outlier_dataset = dataset.iloc[outliers_idx]
    outlier_dataset["isFraud"].value_counts()


