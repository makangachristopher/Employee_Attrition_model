import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data

class Preprocessor:
    def __init__(self, df):
        self.df = df

    def define_target_and_independent_features(self):
        Y = self.df[['Attrition']]
        X = self.df.drop(['Attrition'], axis=1)
        return X, Y

    def remove_features_with_zero_variance(self, X):
        feature_val_count = pd.DataFrame(X.apply(lambda m: m.value_counts().count()))
        feat_level_index = feature_val_count[feature_val_count['uni_level'] > 1].index
        X = X.loc[:, feat_level_index]
        return X

    def separate_feature_into_numerical_and_categorical(self, X):
        num = X.select_dtypes(include='number')
        char = X.select_dtypes(include='object')
        return num, char

    def remove_outliers(self, numerical):
        def outlier_cap(x):
            x = x.clip(lower=x.quantile(0.01))
            x = x.clip(upper=x.quantile(0.99))
            return x

        numerical = numerical.apply(outlier_cap)
        return numerical

    def scale_numerical_features(self, numerical):
        scaler = StandardScaler()
        numerical_scaled = scaler.fit_transform(numerical)
        return numerical_scaled

    def select_k_best_categorical_features(self, catagorical, Y):
        catag_dum = pd.get_dummies(catagorical, drop_first=True)
        selector = SelectKBest(chi2, k=20)
        selector.fit_transform(catag_dum, Y)
        cols = selector.get_support(indices=True)
        X_ca = catag_dum.iloc[:, cols]
        return X_ca

    def join_all_features(self, numerical, X_ca):
        X_all = pd.concat([numerical, X_ca], axis=1, join='inner')
        return X_all

    def oversample_data(self, X_all, Y):
        ros = RandomOverSampler(sampling_strategy='minority', random_state=1)
        X_s, Y_s = ros.fit_resample(X_all, Y)
        return X_s, Y_s

    def split_data(self, X_s, Y_s):
        X_train, x_test, y_train, y_test = train_test_split(X_s.values, Y_s.values, test_size=0.2, random_state=1)
        return X_train, x_test, y_train, y_test

    def torch_geometric_data(self, X_train, x_test, y_train, y_test):
        standard_scaler = StandardScaler()
        X_train = standard_scaler.fit_transform(X_train)
        x_test = standard_scaler.transform(x_test)

        train_graph = Data(
            x=torch.tensor(X_train, dtype=torch.float),
            edge_index=torch.tensor(y_train, dtype=torch.long)
        )

        test_graph = Data(
            x=torch.tensor(x_test, dtype=torch.float),
            edge_index=torch.tensor(y_test, dtype=torch.long)
        )

        return train_graph, test_graph

# Example usage:
# Initialize preprocessor
# preprocessor = Preprocessor(df)

# Call preprocessing methods
# X, Y = preprocessor.define_target_and_independent_features()
# X = preprocessor.remove_features_with_zero_variance(X)
# numerical, categorical = preprocessor.separate_feature_into_numerical_and_categorical(X)
# numerical = preprocessor.remove_outliers(numerical)
# numerical_scaled = preprocessor.scale_numerical_features(numerical)
# X_ca = preprocessor.select_k_best_categorical_features(categorical, Y)
# X_all = preprocessor.join_all_features(numerical_scaled, X_ca)
# X_s, Y_s = preprocessor.oversample_data(X_all, Y)
# X_train, x_test, y_train, y_test = preprocessor.split_data(X_s, Y_s)
# train_graph, test_graph = preprocessor.torch_geometric_data(X_train, x_test, y_train, y_test)
