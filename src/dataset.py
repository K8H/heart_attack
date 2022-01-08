from sklearn.model_selection import train_test_split

import data_load as data
import utils.constants as const


class Dataset(object):

    def __init__(self, ds_name, features, target, gender_sep, logger):
        self.logger = logger
        self.logger.info(f"Start loading {ds_name} dataset")

        ds = None
        if ds_name == 'cleveland':
            ds = data.dataset_clev
        elif ds_name == 'risk_factors':
            ds = data.dataset_risk_fact
            ds.rename(columns = {'male':'sex'}, inplace = True)
        elif ds_name == 'heart_failure':
            ds = data.dataset_hf

        self.generate_dataset(ds, features, target, gender_sep)
        self.logger.info("Dataset loading complete")

    def generate_dataset(self, data_df, features, target, gender_sep):
        data_df.dropna(inplace=True)
        if gender_sep == 'whole':
            self.X = data_df[features]
            self.y = data_df[target]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data_df[features],
                                                                                    data_df[target],
                                                                                    test_size=const.test_size,
                                                                                    random_state=const.random_seed)
        elif gender_sep == 'nosex':
            self.logger.info(f"Number of entries {len(data_df)}")
            features.remove("sex")
            self.X = data_df[features]
            self.y = data_df[target]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data_df[features],
                                                                                    data_df[target],
                                                                                    test_size=const.test_size,
                                                                                    random_state=const.random_seed)
        elif gender_sep == 'male':
            feat_targ = features + [target]
            dataset_male = data_df[data_df['sex'] == 1][feat_targ]
            self.logger.info(f"Number of entries {len(dataset_male)}")
            self.X = data_df[features]
            self.y = data_df[target]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(dataset_male[features],
                                                                                    dataset_male[target],
                                                                                    test_size=const.test_size,
                                                                                    random_state=const.random_seed)
        elif gender_sep == 'female':
            feat_targ = features + [target]
            dataset_female = data_df[data_df['sex'] == 0][feat_targ]
            self.logger.info(f"Number of entries {len(dataset_female)}")
            self.X = data_df[features]
            self.y = data_df[target]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(dataset_female[features],
                                                                                    dataset_female[target],
                                                                                    test_size=const.test_size,
                                                                                    random_state=const.random_seed)
