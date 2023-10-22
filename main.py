import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dataset = pd.read_csv("./data/Fraud.csv")

    sa = StatisticalAnalysis(dataset)
    sp = StatisticalPlots()
    so = StatisticalOutlierDetection()

    me = ModelEvaluation()

    # get NonFraud-Fraud counts
    nonfraud_frad_counts = sa.get_counts("isFraud").values
    print("Normal transactions count: ", f"{nonfraud_frad_counts[0]:,}")
    print("Fraud transactions count: ", f"{nonfraud_frad_counts[1]:,}")
    print("Percentage of frauds: ", nonfraud_frad_counts[1] * 100 / (nonfraud_frad_counts[0] + nonfraud_frad_counts[1]))
    sp.plot_pie(["Not Fraud", "Fraud"], nonfraud_frad_counts, ['g', 'r'])

    # get the smallest account based on activity
    print(sa.get_groupby_size_min("nameOrig"))

    # get the biggest account based on activity
    print(sa.get_groupby_size_max("nameOrig"))

    # analyze types of transactions
    # we can notice a big difference in occurrences for each type of transaction
    print(sa.get_unique_values("type"))
    type_counts = sa.get_counts("type")
    print(type_counts)
    sp.plot_bar(list(range(len(type_counts))), type_counts.values, type_counts.index, color=['r', 'g', 'y', 'b', 'c'])

    #check how many frauds we have for each type of payment
    for pay_type in dataset.type.unique():
        print(pay_type)
        print(dataset.loc[dataset.type == str(pay_type)].isFraud.value_counts())

    # round and convert the key feature so far
    dataset["amount"] = dataset["amount"].round(0).astype(int)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    dataset.loc[dataset.type == "CASH_OUT"]["amount"].plot(kind='kde', ax=axes[0][0]).set_title("CASH_OUT DIST")
    dataset.loc[dataset.type == "PAYMENT"]["amount"].plot(kind='kde', ax=axes[0][1]).set_title("PAYMENT DIST")
    dataset.loc[dataset.type == "CASH_IN"]["amount"].plot(kind='kde', ax=axes[0][2]).set_title("CASH_IN DIST")
    dataset.loc[dataset.type == "TRANSFER"]["amount"].plot(kind='kde', ax=axes[1][0]).set_title("TRANSFER DIST")
    dataset.loc[dataset.type == "DEBIT"]["amount"].plot(kind='kde', ax=axes[1][1]).set_title("DEBIT DIST")

    bins = [0, 1000, 10_000, 100_000, 250_000, 500_000, 1_000_000, 1_000_000_000]
    labels = ["0-1000", "1000-10_000", "10_000-100_000", "100_000-250_000", "250_000-500_000", "500_000-1_000_000",
              "1_000_000-1_000_000_000"]
    dataset['amount_bins'] = pd.cut(dataset['amount'], bins=bins, labels=labels)

    # plot values for amount bins
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    dataset.loc[dataset.type == "CASH_OUT"]["amount_bins"].value_counts().sort_index().plot(ax=axes[0][0]).set_title("CASH_OUT DIST")
    dataset.loc[dataset.type == "PAYMENT"]["amount_bins"].value_counts().sort_index().plot(ax=axes[0][1]).set_title("PAYMENT DIST")
    dataset.loc[dataset.type == "CASH_IN"]["amount_bins"].value_counts().sort_index().plot(ax=axes[0][2]).set_title("CASH_IN DIST")
    dataset.loc[dataset.type == "TRANSFER"]["amount_bins"].value_counts().sort_index().plot(ax=axes[1][0]).set_title("TRANSFER DIST")
    dataset.loc[dataset.type == "DEBIT"]["amount_bins"].value_counts().sort_index().plot(ax=axes[1][1]).set_title("DEBIT DIST")

    # plot distribution shape for amount bins
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    dataset.loc[dataset.type == "CASH_OUT"]["amount_bins"].value_counts().plot(kind='kde', ax=axes[0][0]).set_title("CASH_OUT DIST")
    dataset.loc[dataset.type == "PAYMENT"]["amount_bins"].value_counts().plot(kind='kde', ax=axes[0][1]).set_title("PAYMENT DIST")
    dataset.loc[dataset.type == "CASH_IN"]["amount_bins"].value_counts().plot(kind='kde', ax=axes[0][2]).set_title("CASH_IN DIST")
    dataset.loc[dataset.type == "TRANSFER"]["amount_bins"].value_counts().plot(kind='kde', ax=axes[1][0]).set_title("TRANSFER DIST")
    dataset.loc[dataset.type == "DEBIT"]["amount_bins"].value_counts().plot(kind='kde', ax=axes[1][1]).set_title("DEBIT DIST")

    dataset["zscore_outliers"] = 0

    for pay_type in dataset.type.unique():
        col_zscores = so.zscore(dataset.loc[dataset.type == pay_type]["amount"])
        outliers_idx = so.zscore_outliers(col_zscores)

        print("Z-Score method detected {} outliers inside {} payment type!".format(len(outliers_idx), pay_type))

        dataset.loc[outliers_idx,'zscore_outliers'] = 1

    z_score_f1 = me.f1_score(dataset.isFraud, dataset.zscore_outliers)
    z_score_conf_matrix = me.confusion_matrix(dataset.isFraud, dataset.zscore_outliers)

    print("F1 Score obtained with the Z-Score approach is: ", z_score_f1)
    sp.plot_cofusion_matrix(z_score_conf_matrix)

    #do pandas scatterplot at multivariate analysis
    dataset.loc[dataset.type == "CASH_OUT"].plot.scatter(x = 'newbalanceOrig', y = 'amount')
    dataset.loc[dataset.type == "TRANSFER"].plot.scatter(x = 'newbalanceOrig', y = 'amount')


    #use kbins discretizer