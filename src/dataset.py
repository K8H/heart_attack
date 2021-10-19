from sklearn.model_selection import train_test_split

import data_load as data
import utils.constants as const

from utils.ha_logging import logger


class Dataset(object):

    def __init__(self, ds_name, features, target, gender_sep=None):
        logger.info(f"Start loading {ds_name} dataset")

        ds = None
        if ds_name == 'cleveland':
            ds = data.dataset_clev
        elif ds_name == 'risk_factors':
            ds = data.dataset_risk_fact
            ds.rename(columns = {'male':'sex'}, inplace = True)
        elif ds_name == 'heart_failure':
            ds = data.dataset_hf

        self.generate_dataset(ds, features, target, gender_sep)
        logger.info("Dataset loading complete")

    def generate_dataset(self, data_df, features, target, gender_sep=None):
        data_df.dropna(inplace=True)
        if not gender_sep:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data_df[features],
                                                                                    data_df[target],
                                                                                    test_size=const.test_size,
                                                                                    random_state=const.random_seed)
        elif gender_sep == 'nosex':
            logger.info(f"Number of entries {len(data_df)}")
            features.remove("sex")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data_df[features],
                                                                                    data_df[target],
                                                                                    test_size=const.test_size,
                                                                                    random_state=const.random_seed)
        elif gender_sep == 'male':
            feat_targ = features + [target]
            dataset_male = data_df[data_df['sex'] == 1][feat_targ]
            logger.info(f"Number of entries {len(dataset_male)}")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(dataset_male[features],
                                                                                    dataset_male[target],
                                                                                    test_size=const.test_size,
                                                                                    random_state=const.random_seed)
        elif gender_sep == 'female':
            feat_targ = features + [target]
            dataset_female = data_df[data_df['sex'] == 0][feat_targ]
            logger.info(f"Number of entries {len(dataset_female)}")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(dataset_female[features],
                                                                                    dataset_female[target],
                                                                                    test_size=const.test_size,
                                                                                    random_state=const.random_seed)

    # def heart_failure(self, features, target, gender_sep=None):
    #     data_ris = data.dataset_hf.dropna(inplace=True)
    #     if not gender_sep:
    #         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data.dataset_hf[features],
    #                                                                                 data.dataset_hf[target],
    #                                                                                 test_size=const.test_size,
    #                                                                                 random_state=const.random_seed)
    #     elif gender_sep == 'nosex':
    #         logger.info(f"Number of entries {len(data.dataset_hf)}")
    #         features.remove("sex")
    #         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data.dataset_hf[features],
    #                                                                                 data.dataset_hf[target],
    #                                                                                 test_size=const.test_size,
    #                                                                                 random_state=const.random_seed)
    #     elif gender_sep == 'male':
    #         feat_targ = features + [target]
    #         dataset_male = data.dataset_hf[data.dataset_hf['sex'] == 1][feat_targ]
    #         logger.info(f"Number of entries {len(dataset_male)}")
    #         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(dataset_male[features],
    #                                                                                 dataset_male[target],
    #                                                                                 test_size=const.test_size,
    #                                                                                 random_state=const.random_seed)
    #     elif gender_sep == 'female':
    #         feat_targ = features + [target]
    #         dataset_female = data.dataset_hf[data.dataset_hf['sex'] == 0][feat_targ]
    #         logger.info(f"Number of entries {len(dataset_female)}")
    #         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(dataset_female[features],
    #                                                                                 dataset_female[target],
    #                                                                                 test_size=const.test_size,
    #                                                                                 random_state=const.random_seed)
    #
    # def risk_factors(self, features, target, gender_sep=None):
    #     data_ris = data.dataset_hf.dropna(inplace=True)
    #     if not gender_sep:
    #         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data.dataset_risk_fact[features],
    #                                                                                 data.dataset_risk_fact[target],
    #                                                                                 test_size=const.test_size,
    #                                                                                 random_state=const.random_seed)
    #     elif gender_sep == 'nosex':
    #         logger.info(f"Number of entries {len(data.dataset_risk_fact)}")
    #         features.remove("male")
    #         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data.dataset_risk_fact[features],
    #                                                                                 data.dataset_risk_fact[target],
    #                                                                                 test_size=const.test_size,
    #                                                                                 random_state=const.random_seed)
    #     elif gender_sep == 'male':
    #         feat_targ = features + [target]
    #         dataset_male = data.dataset_risk_fact[data.dataset_risk_fact['male'] == 1][feat_targ]
    #         logger.info(f"Number of entries {len(dataset_male)}")
    #         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(dataset_male[features],
    #                                                                                 dataset_male[target],
    #                                                                                 test_size=const.test_size,
    #                                                                                 random_state=const.random_seed)
    #
    #     elif gender_sep == 'female':
    #         feat_targ = features + [target]
    #         dataset_female = data.dataset_risk_fact[data.dataset_risk_fact['male'] == 0][feat_targ]
    #         logger.info(f"Number of entries {len(dataset_female)}")
    #         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(dataset_female[features],
    #                                                                                 dataset_female[target],
    #                                                                                 test_size=const.test_size,
    #                                                                                 random_state=const.random_seed)
    #
    # def cleveland(self, features, target, gender_sep=None):
    #     if not gender_sep:
    #         logger.info(f"Number of entries {len(data.dataset_clev)}")
    #         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data.dataset_clev[features],
    #                                                                                 data.dataset_clev[target],
    #                                                                                 test_size=const.test_size,
    #                                                                                 random_state=const.random_seed)
    #
    #     elif gender_sep == 'nosex':
    #         logger.info(f"Number of entries {len(data.dataset_clev)}")
    #         features.remove("sex")
    #         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data.dataset_clev[features],
    #                                                                                 data.dataset_clev[target],
    #                                                                                 test_size=const.test_size,
    #                                                                                 random_state=const.random_seed)
    #     elif gender_sep == 'male':
    #         feat_targ = features + [target]
    #         dataset_male = data.dataset_clev[data.dataset_clev['sex'] == 1][feat_targ]
    #         logger.info(f"Number of entries {len(dataset_male)}")
    #         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(dataset_male[features],
    #                                                                                 dataset_male[target],
    #                                                                                 test_size=const.test_size,
    #                                                                                 random_state=const.random_seed)
    #
    #     elif gender_sep == 'female':
    #         feat_targ = features + [target]
    #         dataset_female = data.dataset_clev[data.dataset_clev['sex'] == 0][feat_targ]
    #         logger.info(f"Number of entries {len(dataset_female)}")
    #         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(dataset_female[features],
    #                                                                                 dataset_female[target],
    #                                                                                 test_size=const.test_size,
    #                                                                                 random_state=const.random_seed)
