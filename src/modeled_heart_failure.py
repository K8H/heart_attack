import matplotlib.pyplot as plt
import seaborn as sns
import shap
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, accuracy_score, plot_roc_curve

import utils.constants as const
from src.dataset import Dataset
from utils.ha_logging import logger

experiment_id = 'hf001'
file_name = "experiments/configs/heart_failure.yaml"

with open(file_name, "r") as yamlfile:
    data = yaml.load(yamlfile, Loader=yaml.FullLoader)
    logger.info(f"Read successful file {file_name}")
    targ = data[experiment_id]['target']
    dataset = Dataset('heart_failure', data[experiment_id]['features'], data[experiment_id]['target'])
    dataset_nosex = Dataset('heart_failure', data[experiment_id]['features'], data[experiment_id]['target'], "nosex")
    sex = dataset.X_train["sex"].to_numpy()

    clf = RandomForestClassifier(max_depth=data[experiment_id]['max_depth'], random_state=const.random_seed)
    clf.fit(dataset.X_train, dataset.y_train)
    y_pred = clf.predict(dataset.X_test)
    accuracy_sc = accuracy_score(dataset.y_test.values, y_pred)

    roc_auc = roc_curve(dataset.y_test.values, y_pred)
    plot_roc_curve(clf, dataset.X_test.values, dataset.y_test.values)
    plt.savefig("results/plots/roc_curve.png", bbox_inches='tight', pad_inches=0)

    logger.info(f"accuracy {accuracy_sc}")

    clf_nosex = RandomForestClassifier(max_depth=data[experiment_id]['max_depth'], random_state=const.random_seed)
    clf_nosex.fit(dataset_nosex.X_train, dataset_nosex.y_train)
    y_pred_nosex = clf_nosex.predict(dataset_nosex.X_test)
    accuracy_sc_nosex = accuracy_score(dataset_nosex.y_test.values, y_pred_nosex)

    logger.info(f"accuracy nosex {accuracy_sc_nosex}")

    # shap part
    shap_values = shap.TreeExplainer(clf).shap_values(dataset.X_train)
    shap_values_nosex = shap.TreeExplainer(clf_nosex).shap_values(dataset_nosex.X_train)

    xmin = -0.8
    xmax = 0.8

    shap.summary_plot(shap_values, dataset.X_train, plot_type="bar", show=False)
    plt.savefig("results/plots/heart_failure_shap.png", bbox_inches='tight', pad_inches=0)

    shap.dependence_plot("sex", shap_values[1], dataset.X_train, xmin=xmin, xmax=xmax, show=False)
    plt.savefig("results/plots/heart_failure_shap_sex_depend.png", bbox_inches='tight', dpi=600)

    shap.group_difference_plot(shap_values[1], sex, dataset.X_train.columns, xmin=xmin, xmax=xmax, show=False)
    plt.savefig("results/plots/heart_failure_shap_sex_group_diff.png", bbox_inches='tight', dpi=600)

    shap.group_difference_plot(shap_values_nosex[1], sex, dataset_nosex.X_train.columns, xmin=xmin, xmax=xmax, show=False)
    plt.savefig("results/plots/heart_failure_shap_sex_group_diff_nosex_features.png", bbox_inches='tight', dpi=600)

    shap.group_difference_plot(shap_values_nosex[1].sum(1), sex, xmin=xmin, xmax=xmax, show=False)
    plt.savefig("results/plots/heart_failure_shap_sex_group_diff_nosex.png", bbox_inches='tight', dpi=600)

    # male
    logger.info("Prediction male ---------")
    dataset = Dataset('heart_failure', data[experiment_id]['features'], data[experiment_id]['target'], gender_sep='male')
    clf = RandomForestClassifier(max_depth=data[experiment_id]['max_depth'], random_state=const.random_seed)
    clf.fit(dataset.X_train, dataset.y_train)

    y_pred = clf.predict(dataset.X_test)
    roc_auc_male = roc_curve(dataset.y_test.values, y_pred)
    accuracy_sc_male = accuracy_score(dataset.y_test.values, y_pred)
    logger.info(f"accuracy {accuracy_sc_male}")
    plot_roc_curve(clf, dataset.X_test.values, dataset.y_test.values)
    plt.savefig("results/plots/roc_curve_male.png", bbox_inches='tight', dpi=600)

    # shap part males
    shap_values = shap.TreeExplainer(clf).shap_values(dataset.X_train)
    shap.summary_plot(shap_values, dataset.X_train, plot_type="bar", show=False)
    plt.savefig("results/plots/heart_failure_shap_male.png", bbox_inches='tight', dpi=600)

    # female
    logger.info("Prediction female ---------")
    dataset = Dataset('heart_failure', data[experiment_id]['features'], data[experiment_id]['target'], gender_sep='female')
    clf = RandomForestClassifier(max_depth=data[experiment_id]['max_depth'], random_state=const.random_seed)
    clf.fit(dataset.X_train, dataset.y_train)

    y_pred = clf.predict(dataset.X_test)
    roc_auc_female = roc_curve(dataset.y_test.values, y_pred)
    accuracy_sc_female = accuracy_score(dataset.y_test.values, y_pred)
    logger.info(f"accuracy {accuracy_sc_female}")
    plot_roc_curve(clf, dataset.X_test.values, dataset.y_test.values)
    plt.savefig("results/plots/roc_curve_female.png", bbox_inches='tight', dpi=600)

    # shap part males
    shap_values = shap.TreeExplainer(clf).shap_values(dataset.X_train)
    shap.summary_plot(shap_values, dataset.X_train, plot_type="bar", show=False)
    plt.savefig("results/plots/heart_failure_shap_female.png", bbox_inches='tight', dpi=600)


    f, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(dataset.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
    plt.show()