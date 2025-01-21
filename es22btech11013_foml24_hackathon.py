#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline  
from imblearn.combine import SMOTEENN


pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.max_rows', None)


def make_predictions(test_fname, predictions_fname):
    df = pd.read_csv("train.csv")
    df.head(90)
    print(df.shape)
    X_test = pd.read_csv(test_fname)
    # df.head(20)


    features_ordinal = ['FarmClassification', 'NumberGreenHouses']
    features_year = ['TaxOverdueYear', 'ValuationYear', 'FieldEstablishedYear']

    features_categorical = []
    for feature in df.columns:
        if (df[feature].nunique() < 50) and (feature not in (features_ordinal)) and (feature not in features_year) and(feature != 'Target'):
            features_categorical.append(feature)

    df['TaxOverdueYear'].fillna('missing', inplace=True)
    df['ValuationYear'] = 2020 - df['ValuationYear']
    df['FieldEstablishedYear'] = 2020 - df['FieldEstablishedYear']

    features_categorical.append('TaxOverdueYear')

    features_numerical = []
    for feature in df.columns:
        if (df[feature].nunique() >= 50) and (feature not in (features_ordinal)) and (feature not in features_year) and (feature != 'Target'):
            features_numerical.append(feature)
    features_numerical.append('ValuationYear')
    features_numerical.append('FieldEstablishedYear')
    features = features_numerical + features_categorical + features_ordinal
    print(len(features))
    print(len(features_categorical))

    features_drop = ['FarmingCommunityId', 'AgriculturalPostalZone', 'OtherZoningCode', 'AgricultureZoningCode']
    features = [feature for feature in features if feature not in features_drop]
    df = df.drop(features_drop, axis = 1)
    features_categorical = [feature for feature in features_categorical if feature not in features_drop]
    features_numerical = [feature for feature in features_numerical if feature not in features_drop]


    features_num_not_cat = ['WaterAccessPointsCalc', 'WaterAccessPoints', 'StorageAndFacilityCount', 'MainIrrigationSystemCount', 'FarmingUnitCount']

    features_categorical = [feature for feature in features_categorical if feature not in features_num_not_cat]

    features_numerical.extend(features_num_not_cat)
    print(len(features))
    features_cat_nan = []
    features_cat_drop = []
    for feature in features_categorical:
        num_nans = df[feature].isnull().mean()
        if num_nans > 0.90:
            features_cat_nan.append(feature)

    features_drop = features_cat_nan
    features = [feature for feature in features if feature not in features_drop]
    df = df.drop(features_drop, axis = 1)
    features_categorical = [feature for feature in features_categorical if feature not in features_drop]
    features_numerical = [feature for feature in features_numerical if feature not in features_drop]
    print(len(features))
    features_num_nan = []
    features_num_drop = []
    for feature in features_numerical:
        num_nans = df[feature].isnull().mean()
        if num_nans > 0.90:
            features_num_nan.append(feature)


    features_drop = features_num_nan
    features = [feature for feature in features if feature not in features_drop]
    df = df.drop(features_drop, axis = 1)
    features_categorical = [feature for feature in features_categorical if feature not in features_drop]
    features_numerical = [feature for feature in features_numerical if feature not in features_drop]
    print(len(features))
    X_test = X_test[features]
    print(df.shape)
    columns_not_in_features = set(df.columns) - set(features)

    #check if the columns are mismatching
    print("Columns in df but not in features:", columns_not_in_features)
    print("Columns in df:", list(df.columns))
    print(len(features_categorical))
    print(len(features_numerical))
    print(len(features_ordinal))
    print(df.shape)

    for feature in features_categorical:
        print(f"{feature}: ", df[feature].nunique())

    print(df.shape)
    print(X_test.shape)


    df = pd.get_dummies(df, columns=features_categorical, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=features_categorical, drop_first=True)
    X_train = df.drop(columns=['Target'])


    x_test_columns = set(X_test.columns)
    x_train_columns_set = set(X_train.columns)

    all_features_covered = x_test_columns == x_train_columns_set
    print(f'Features of C_train and X_test are same: {all_features_covered}')

    features_only_in_features_tot = x_test_columns - x_train_columns_set
    features_only_in_x_train = x_train_columns_set - x_test_columns

    print("Features in x_test but not in X_train:", features_only_in_features_tot)
    print("Features in X_train but not in x_test:", features_only_in_x_train)


    X_test_columns = X_test.columns
    X_train = X_train.reindex(columns=X_test.columns, fill_value=0)

    y_train = df['Target']



    threshold = 0.003
    X_train_new = X_train.drop('UID', axis=1).copy()
    X_test_new = X_test.drop('UID', axis=1).copy()

    features_categorical = [feature for feature in X_train_new.columns if feature not in features_numerical]
    numerical_features = [feature for feature in features_numerical if feature in X_train_new.columns]
    categorical_features = [feature for feature in features_categorical if feature in X_train_new.columns]

    features_tot = numerical_features + categorical_features
    features_tot_set = set(features_tot)
    x_train_columns_set = set(X_train_new.columns)

    all_features_covered = features_tot_set == x_train_columns_set
    print(f'Are all features covered in X_train: {all_features_covered}')

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)


    features_only_in_features_tot = features_tot_set - x_train_columns_set
    features_only_in_x_train = x_train_columns_set - features_tot_set
    print("Features in features_tot but not in X_train:", features_only_in_features_tot)
    print("Features in X_train but not in features_tot:", features_only_in_x_train)

    df_combined = X_train_new.copy()
    df_combined['Target'] = y_train_encoded

    correlation_matrix = df_combined.corr()

    feature_correlations = correlation_matrix['Target'].drop('Target')

    if threshold == 0:
        selected_features = feature_correlations.index.tolist()
    else:
        selected_features = feature_correlations[feature_correlations.abs() > threshold].index.tolist()

    print(f"Selected {len(selected_features)} features with |correlation| > {threshold}")
    X_train_threshold = X_train_new[selected_features].copy()
    X_test_threshold = X_test_new[selected_features].copy()

    numerical_features_threshold = [feature for feature in numerical_features if feature in selected_features]
    categorical_features_threshold = [feature for feature in categorical_features if feature in selected_features]

    numerical_transformer = SimpleImputer(strategy='median')
    categorical_transformer = SimpleImputer(strategy='most_frequent')

    preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_features_threshold),('cat', categorical_transformer, categorical_features_threshold)])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('resampler', SMOTEENN(random_state=12)),
    ('classifier', RandomForestClassifier(random_state=12, n_jobs=-1, class_weight='balanced', n_estimators=1000, min_samples_split=2, min_samples_leaf=4, max_features='log2', max_depth=None))
    ])

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    print("training the final model on the entire training dataset")
    pipeline.fit(X_train_threshold, y_train_encoded)
    print("final training done")

    print("predictions on the test set")
    predictions = pipeline.predict(X_test_threshold)

    predicted_labels = label_encoder.inverse_transform(predictions)

    submission_df = pd.DataFrame({'UID': X_test['UID'],'Target': predicted_labels})

    submission_df.to_csv(predictions_fname, index=False)
    print("predictions saved to submission.csv")


# In[ ]:


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=str, help='file path of train.csv')
    parser.add_argument("--test-file", type=str, help='file path of test.csv')
    parser.add_argument("--predictions-file", type=str, help='save path of predictions')
    args = parser.parse_args()
    make_predictions(args.test_file, args.predictions_file)

