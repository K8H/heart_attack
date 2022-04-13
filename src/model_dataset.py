import os
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import shap
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, accuracy_score, plot_roc_curve

import utils.constants as const
from src.dataset import Dataset
from src.utils.lib import if_not_mkdir
from utils.ha_logging import Logger


class Module(object):
    xmin = -0.4
    xmax = 0.4

    def __init__(self, dataset_name, experiment_id):
        self.root_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
        self.log_dir = os.path.join(self.root_dir, f'results/logs/{dataset_name}')
        self.plot_dir = os.path.join(self.root_dir, f'results/plots/{dataset_name}')
        if_not_mkdir(self.log_dir)
        if_not_mkdir(self.plot_dir)
        self.logger = Logger(os.path.join(self.log_dir, f'{experiment_id}')).logger
        self.dataset_name = dataset_name
        self.experiment_id = experiment_id
        self.data = None

        # whole dataset
        self.logger.info('********************************* Whole dataset')
        self.gender_sep = 'whole'
        dataset = self.get_dataset()
        sex = dataset.X_train["sex"].to_numpy()     # TODO try to put that into plot function
        clf, y_pred = self.model(dataset)
        self.plot(dataset, clf, y_pred, sex)

        # no sex
        self.logger.info('********************************* Whole dataset without gender feature')
        self.gender_sep = 'nosex'
        dataset = self.get_dataset()
        clf, y_pred = self.model(dataset)
        self.plot(dataset, clf, y_pred, sex)

        # male
        self.logger.info('********************************* Male part of the dataset')
        self.gender_sep = 'male'
        dataset = self.get_dataset()
        clf, y_pred = self.model(dataset)
        self.plot(dataset, clf, y_pred, sex)

        # female
        self.logger.info('********************************* Female part of the dataset')
        self.gender_sep = 'female'
        dataset = self.get_dataset()
        clf, y_pred = self.model(dataset)
        self.plot(dataset, clf, y_pred, sex)

    def get_dataset(self):
        with open(f"../experiments/configs/{self.dataset_name}.yaml", "r") as yamlfile:
            self.data = yaml.load(yamlfile, Loader=yaml.FullLoader)
            self.logger.info(f"Read successful file for dataset {self.dataset_name}")
            targ = self.data[self.experiment_id]['target']
            dataset = Dataset(self.dataset_name,
                              self.data[self.experiment_id]['features'],
                              self.data[self.experiment_id]['target'],
                              self.gender_sep,
                              self.logger)
            return dataset

    def model(self, dataset):
        clf = RandomForestClassifier(max_depth=self.data[self.experiment_id]['max_depth'], random_state=const.random_seed)
        clf.fit(dataset.X_train, dataset.y_train)
        y_pred = clf.predict(dataset.X_test)
        accuracy_sc = accuracy_score(dataset.y_test.values, y_pred)

        self.logger.info(f"accuracy {accuracy_sc}")

        return clf, y_pred

    def plot(self, dataset, clf, y_pred, sex):
        roc_auc = roc_curve(dataset.y_test.values, y_pred)
        plot_roc_curve(clf, dataset.X_test.values, dataset.y_test.values)
        plt.savefig(os.path.join(self.plot_dir, f"{self.gender_sep}_roc_curve.png"),
                    bbox_inches='tight',
                    pad_inches=0)   # TPDP maybe for man instead pad_inches -> dpi=600

        # shap part
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(dataset.X_train)

        shap.summary_plot(shap_values[0], dataset.X_train, max_display=5, show=False)
        plt.savefig(os.path.join(self.plot_dir, f'{self.gender_sep}_shap_summary.png'),
                    bbox_inches='tight',
                    pad_inches=0)

        f, ax = plt.subplots(figsize=(15, 15))
        sns.heatmap(dataset.X.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
        plt.savefig(os.path.join(self.plot_dir, f'{self.gender_sep}_heatmap_features.png'),
                    bbox_inches='tight', dpi=600)
        plt.close(f)

        if self.gender_sep == 'whole':
            X_train_cat = dataset.X_train.copy()
            X_train_cat['sex'] = X_train_cat['sex'].replace({0: 'female', 1: 'male'})
            shap.dependence_plot("sex", shap_values[0], X_train_cat, show=False)
            plt.savefig(os.path.join(self.plot_dir, f'{self.gender_sep}_shap_dependence_plot.png'),
                        bbox_inches='tight', dpi=600)

            glabel_feat = "Demographic parity difference\nof model output for female vs. male\nby feature"
            shap.group_difference_plot(shap_values[0], sex, dataset.X_train.columns, xmin=self.xmin, xmax=self.xmax,
                                       xlabel=glabel_feat, show=False)
            plt.savefig(os.path.join(self.plot_dir, f'{self.gender_sep}_shap_group_diff_plot.png'),
                        bbox_inches='tight', dpi=600)

            self.statistics(dataset)

        if self.gender_sep == 'nosex':
            glabel_feat = "Demographic parity difference\nof model output for female vs. male\nby feature"
            shap.group_difference_plot(shap_values[0], sex, dataset.X_train.columns, xmin=self.xmin, xmax=self.xmax,
                                       xlabel=glabel_feat, show=False)
            plt.savefig(os.path.join(self.plot_dir, f'{self.gender_sep}_shap_group_diff_plot_features.png'),
                        bbox_inches='tight', dpi=600)

            glabel = "Demographic parity difference\nof model output for female vs. male"
            shap.group_difference_plot(shap_values[0].sum(1), sex, xmin=self.xmin, xmax=self.xmax, xlabel=glabel,
                                       show=False)
            plt.savefig(os.path.join(self.plot_dir, f'{self.gender_sep}_shap_group_diff_plot.png'),
                        bbox_inches='tight', dpi=600)

    def statistics(self, dataset):
        man = dataset.X['sex'][(dataset.X['sex'] == 1)]
        woman = dataset.X['sex'][(dataset.X['sex'] == 0)]

        m = (man.count() * 100) / len(dataset.X)
        w = (woman.count() * 100) / len(dataset.X)

        labels = 'Men', 'Women'
        sizes = [m, w]
        explode = (0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title("% Men & Women", bbox={'facecolor': '0.8', 'pad': 5})
        plt.savefig(os.path.join(self.plot_dir, f'{self.gender_sep}_female_male_piechart.png'),
                    bbox_inches='tight', dpi=600)

        AM = dataset.X[['age', 'sex']][
            (dataset.X['sex'] == 0) & (dataset.X['age'] >= 45) & (dataset.X['age'] <= 74)]

        labels = 'Hypotese age (45-74 years)', 'Out Hypotese age'
        sizes = [(AM.count()[0] * 100) / woman.count(), ((woman.count() - AM.count()[0]) * 100) / woman.count()]
        explode = (0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title("%Woman in hypotese age ", bbox={'facecolor': '0.8', 'pad': 5})
        plt.savefig(os.path.join(self.plot_dir, f'{self.gender_sep}_female_age_piechart.png'),
                    bbox_inches='tight', dpi=600)


if __name__=='__main__':
    Module('cleveland', 'clev001')
    Module('heart_failure', 'hf001')
    Module('risk_factors', 'risk001')
