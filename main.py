import random
import shap

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from gml import *
from deepl import *

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
    print("Minimum number of transactions at client/account level: ")
    print(sa.get_groupby_size_min("nameOrig"))

    # get the biggest account based on activity
    print("Maximum number of transactions at client/account level: ")
    print(sa.get_groupby_size_max("nameOrig"))

    # analyze types of transactions
    # we can notice a big difference in occurrences for each type of transaction
    print(sa.get_unique_values("type"))
    type_counts = sa.get_counts("type")
    print(type_counts)
    sp.plot_bar(list(range(len(type_counts))), type_counts.values, type_counts.index, color=['r', 'g', 'y', 'b', 'c'])

    # check how many frauds we have for each type of payment
    for pay_type in dataset.type.unique():
        print("PAY type: ", str(pay_type))
        print(dataset.loc[dataset.type == str(pay_type)].isFraud.value_counts())
        print()

    # round and convert the key feature so far
    dataset["amount"] = dataset["amount"].round(0).astype(int)

    # plotting transaction type - amount relations
    cashout_mean_amt = dataset.loc[dataset.type == "CASH_OUT"]["amount"].mean().round()
    payment_mean_amt = dataset.loc[dataset.type == "PAYMENT"]["amount"].mean().round()
    cashin_mean_amt = dataset.loc[dataset.type == "CASH_IN"]["amount"].mean().round()
    transfer_mean_amt = dataset.loc[dataset.type == "TRANSFER"]["amount"].mean().round()
    debit_mean_amt = dataset.loc[dataset.type == "DEBIT"]["amount"].mean().round()

    sp.plot_bar(bars_coord=[1, 2, 3, 4, 5],
                heights=[cashout_mean_amt, payment_mean_amt, cashin_mean_amt, transfer_mean_amt, debit_mean_amt],
                x_labels=["CASHOUT MAMT", "PAYMENT MAMT", "CASH_IN MAMT", "TRANSFER MAMT", "DEBIT MAMT"],
                color=['r', 'g', 'y', 'b', 'c'])

    # plotting distributions of amounts based on transactional type to better understand viable techniques
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

    # creating a dataframe that will hold all the results
    df_results = pd.DataFrame(columns=['F1-Score', 'TP', 'Total Alerts (TP+FP)'])

    # ===================== Z-SCORES APPROACH =====================

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

    # storing results
    df_results.loc['z-score'] = [z_score_f1, z_score_conf_matrix[1,1], (z_score_conf_matrix[1,1] + z_score_conf_matrix[0,1])]

    # ===================== MAHALANOBIS DISTANCE APPROACH =====================

    # we want to add more context
    # pandas scatterplot at multivariate analysis
    dataset.loc[dataset.type == "CASH_OUT"].plot.scatter(x = 'newbalanceOrig', y = 'amount')
    dataset.loc[dataset.type == "TRANSFER"].plot.scatter(x = 'newbalanceOrig', y = 'amount')

    transfer_mahalanobis_dists = so.mahalanobis_distance(dataset.loc[dataset.type == "TRANSFER"][["amount", "newbalanceOrig"]])

    dataset["mah_dist_outliers"] = 0

    for pay_type in dataset.type.unique():
        curr_dataset = dataset.loc[dataset.type == pay_type][["amount", "newbalanceOrig"]]
        pt_mahalanobis_dists = so.mahalanobis_distance(curr_dataset)
        mah_outliers_idx = so.mahalanobis_dist_outliers(curr_dataset, pt_mahalanobis_dists)

        print("Mahalanobis-Distance method detected {} outliers inside {} payment type!".format(len(mah_outliers_idx), pay_type))

        dataset.loc[mah_outliers_idx, 'mah_dist_outliers'] = 1

    mah_dist_f1 = me.f1_score(dataset.isFraud, dataset.mah_dist_outliers)
    mah_dist_conf_matrix = me.confusion_matrix(dataset.isFraud, dataset.mah_dist_outliers)

    print("F1 Score obtained with the Mahalanobis Distance approach is: ", mah_dist_f1)
    sp.plot_cofusion_matrix(mah_dist_conf_matrix)

    # storing results
    df_results.loc['mah-dist'] = [mah_dist_f1, mah_dist_conf_matrix[1, 1],
                                 (mah_dist_conf_matrix[1, 1] + mah_dist_conf_matrix[0, 1])]

    # ===================== FEATURE ENGINEERING =====================

    features_df = dataset[["step", "type", "amount", "oldbalanceOrg",
                           "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]].copy(deep=True)

    fe = FeatureEngineering(features_df)

    fe.time_step_encoder()
    fe.one_hot_encoder('type')
    fe.kbins_discretizer("amount")
    fe.kbins_discretizer("oldbalanceOrg")
    fe.kbins_discretizer("newbalanceOrig")
    fe.kbins_discretizer("oldbalanceDest")
    fe.kbins_discretizer("newbalanceDest")
    vec_features_df = fe.df

    scaler = MinMaxScaler(feature_range=(-1, 1))
    vec_features_df[['amount_bin', 'oldbalanceOrg_bin', 'newbalanceOrig_bin',
                     'oldbalanceDest_bin', 'newbalanceDest_bin']] = (
        scaler.fit_transform(vec_features_df[['amount_bin', 'oldbalanceOrg_bin', 'newbalanceOrig_bin',
                                              'oldbalanceDest_bin', 'newbalanceDest_bin']]))

    # ===================== CLUSTERING APPROACH =====================

    # eps 0.5 -> 0.3 - > 0.2 -> 0.1
    # min_s 5 -> 10 -> 15
    cm = ClusteringModel(eps=0.5, min_samples=5)
    cm.fit(vec_features_df)

    print(np.unique(cm.labels, return_counts=True))

    dataset["dbscan_outliers"] = 0
    dbscan_outliers_idx = np.where(cm.labels == -1)[0]
    dataset.loc[dbscan_outliers_idx, 'dbscan_outliers'] = 1

    dbscan_f1 = me.f1_score(dataset.isFraud, dataset.dbscan_outliers)
    dbscan_conf_matrix = me.confusion_matrix(dataset.isFraud, dataset.dbscan_outliers)

    print("F1 Score obtained with the DBSCAN CLUSTERING approach is: ", dbscan_f1)
    sp.plot_cofusion_matrix(dbscan_conf_matrix)

    # storing results
    df_results.loc['dbscan'] = [dbscan_f1, dbscan_conf_matrix[1, 1],
                                (dbscan_conf_matrix[1, 1] + dbscan_conf_matrix[0, 1])]

    # ===================== ISOLATION FOREST APPROACH =====================

    # cont auto -> 0.1 -> 0.01

    iso_forest = IsolationForestModel(contamination='auto', random_state=22)
    iso_forest.fit(vec_features_df)
    if_pred = iso_forest.predict(vec_features_df)

    print(np.unique(if_pred, return_counts=True))

    dataset["iso_forest_outliers"] = 0
    isofor_outliers_idx = np.where(if_pred == -1)[0]
    dataset.loc[isofor_outliers_idx, 'iso_forest_outliers'] = 1

    isofor_f1 = me.f1_score(dataset.isFraud, dataset.iso_forest_outliers)
    isofor_conf_matrix = me.confusion_matrix(dataset.isFraud, dataset.iso_forest_outliers)

    print("F1 Score obtained with the ISOLATION FOREST approach is: ", isofor_f1)
    sp.plot_cofusion_matrix(isofor_conf_matrix)

    # storing results
    df_results.loc['isofor'] = [isofor_f1, isofor_conf_matrix[1, 1],
                                (isofor_conf_matrix[1, 1] + isofor_conf_matrix[0, 1])]

    # ===================== AUTO-ENCODER APPROACH =====================

    non_fraud_idx = dataset.loc[dataset["isFraud"] == 0].index.tolist()
    amount_to_sample = int(np.round((2/3) * len(vec_features_df.iloc[non_fraud_idx])))

    random.seed(a=22)
    ae_X_nof_idx = random.sample(non_fraud_idx, amount_to_sample)
    ae_Y_nof_idx = list(set(non_fraud_idx) - set(ae_X_nof_idx))
    ae_Y_f_idx = dataset.loc[dataset["isFraud"] == 1].index.tolist()

    ae_X = vec_features_df.iloc[ae_X_nof_idx]
    ae_Y = vec_features_df.iloc[ae_Y_nof_idx + ae_Y_f_idx]

    ae = AutoEncoder()

    # define and fit the model
    # ae.define_model()
    # ae.fit(ae_X, ae_Y)

    ae.load_ae_model("./models/AE_AD.keras")

    # thrld 0.5 -> 0.2 -> 0.01 -> 0.001
    ae_outliers_idx = ae.anomaly_detection(ae_Y, threshold=0.5)
    dataset["ae_outliers"] = 0
    dataset.loc[ae_outliers_idx, 'ae_outliers'] = 1

    ae_f1 = me.f1_score(dataset.isFraud, dataset.ae_outliers)
    ae_conf_matrix = me.confusion_matrix(dataset.isFraud, dataset.ae_outliers)

    print("F1 Score obtained with the AUTO ENCODER approach is: ", ae_f1)
    sp.plot_cofusion_matrix(ae_conf_matrix)

    # storing results
    df_results.loc['ae'] = [ae_f1, ae_conf_matrix[1, 1],
                                (ae_conf_matrix[1, 1] + ae_conf_matrix[0, 1])]

    # ===================== AUDIT =====================

    # data assumptions
    subset_of_nameorig = dataset["nameOrig"].unique()[:10]
    df_data_assumpt = dataset.loc[dataset["nameOrig"].isin(subset_of_nameorig)]

    df_data_assumpt.loc[0, "anual_expected_volume"] = 90_000
    df_data_assumpt.loc[1, "anual_expected_volume"] = 15_000
    df_data_assumpt.loc[3, "anual_expected_volume"] = 20_000
    df_data_assumpt.loc[5, "anual_expected_volume"] = 1_000
    df_data_assumpt.loc[7, "anual_expected_volume"] = 1_000_000
    df_data_assumpt.loc[8, "anual_expected_volume"] = 70_000

    df_data_assumpt['anual_expected_volume'] = df_data_assumpt['anual_expected_volume'].fillna(10_000)

    # mode assumptions
    # slides

    # explainable
    outliers_to_be_explained = ae_Y[ae_Y.index.isin(ae_outliers_idx)]

    explainer = shap.Explainer(ae.model, ae_X.values)
    shap_values_anomalies = explainer.shap_values(outliers_to_be_explained.values)

    # create explainer column
    outliers_to_be_explained["shap_explainer"] = ""

    for i, shap_values in enumerate(shap_values_anomalies):
        # print(f"Anomaly {i + 1} - Top 3 Most Important Features:")
        # Find the indices of the top 3 features with the highest absolute SHAP values
        top_feature_indices = np.argsort(-np.abs(shap_values))[:3]
        # print(top_feature_indices)

        unique_features = []
        for feat in top_feature_indices:
            for val in feat:
                if val not in unique_features:
                    unique_features.append(val)
                    break
            if len(unique_features) == 3:
                break

        # print(unique_features)
        unique_most_imp_features = ae_Y.columns[unique_features].values.tolist()
        # print(unique_most_imp_features)

        curr_index = outliers_to_be_explained.iloc[i].name
        outliers_to_be_explained.loc[curr_index, 'shap_explainer'] = ",".join(unique_most_imp_features)

    # ===================== BUSINESS =====================

    # traceback explainer to the original data
    dataset["Outlier Flag"] = 0
    dataset.loc[outliers_to_be_explained.index.values, "Outlier Flag"] = 1

    dataset["Explainer"] = ""
    dataset.loc[outliers_to_be_explained.index.values, "Explainer"] = outliers_to_be_explained['shap_explainer']

    # create explainer story
    dataset["Explainer"] = dataset["Explainer"].apply(lambda x: x.split(","))
    dataset["Explainer"] = dataset["Explainer"].apply(lambda x: list(set([el.split("_")[0] for el in x])))
    print(dataset["Explainer"].value_counts())

    dataset["Explainer"] = (
        dataset[["Outlier Flag", "Explainer"]]
        .apply(lambda x: "The ML model considered this transaction suspicious because of the following attributes: " +
                         ",".join(x["Explainer"]) if x["Outlier Flag"] == 1 else x["Explainer"], axis=1))

    print(dataset["Explainer"].value_counts())

    # too many alerts
    # re-run AE with thrld 0.005 -> 0.002