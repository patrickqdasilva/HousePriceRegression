# These are the functions that I will use to perform the House Price Regression
# Written by Patrick Da Silva
os.link(https://www.linkedin.com/in/patrick-da-silva-871833225/)

import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.feature_selection import mutual_info_regression
from sklearn.cluster import KMeans

# Content

# Categorical Feature Encoding Function
# Mutual Information Score Function 
# Feature Building Functions


####    Categorical Feature Encoding Function    ####

def categorical_feature_encoding(house_data):
    """ Corrects the category encoding for this specific dataset given a DataFrame """
    ### Preprocessing for nominal categorical features ###
    nominal_features = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LandContour', 'LotConfig', 'Neighborhood', 
                        'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 
                        'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'GarageType', 'MiscFeature', 
                        'SaleType', 'SaleCondition']
    # Apply desired characteristics to the house_data dataframe
    for feature in nominal_features:
        house_data[feature] = house_data[feature].astype('category')
        #add 'None' level to nominal features in case features have missing data
        if 'None' not in house_data[feature].cat.categories:
            house_data[feature] = house_data[feature].cat.add_categories('None')

    ### Preprocessing for ordinal categorical features ###

    # These are the shared characteristics of several ordinal features
    five_level = ['Po', 'Fa', 'TA', 'Gd', 'Ex']
    ten_level  = list(range(1,11))

    # Ascribing ordinal features with their levels
    ordinal_levels = {
        'OverallQual': ten_level,
        'OverallCond': ten_level,
        'ExterQual': five_level,
        'ExterCond': five_level,
        'BsmtQual': five_level,
        'BsmtCond': five_level,
        'HeatingQC': five_level,
        'KitchenQual': five_level,
        'FireplaceQu': five_level,
        'GarageQual': five_level,
        'GarageCond': five_level,
        'PoolQC': five_level,
        'LotShape': ['Reg', 'IR1', 'IR2', 'IR3'],
        'LandSlope': ['Sev', 'Mod', 'Gtl'],
        'BsmtExposure': ['No', 'Mn', 'Av', 'Gd'],
        'BsmtFinType1': ['Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
        'BsmtFinType2': ['Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
        'Functional': ['Sal', 'Sev', 'Maj1', 'Maj2', 'Mod', 'Min2', 'Min1', 'Typ'],
        'GarageFinish': ['Unf', 'RFn', 'Fin'],
        'PavedDrive': ['N', 'P', 'Y'],
        'Utilities': ['NoSeWa', 'NoSewr', 'AllPub'],
        'CentralAir': ['N', 'Y'],
        'Electrical': ['Mix', 'FuseP', 'FuseF', 'FuseA', 'SBrkr'],
        'Fence': ['MnWw', 'GdWo', 'MnPrv', 'GdPrv'],
    }

    # Employ dictionary comprehension to add 'None' level to ordinal features in case features have missing data
    ordinal_levels = {feature: ['None'] + level for feature, level in ordinal_levels.items()}

    # Apply the new characteristics to the house_data dataframe
    for feature, level in ordinal_levels.items():
        house_data[feature] = house_data[feature].astype(CategoricalDtype(level, ordered=True))

    return house_data


####    Mutual Information Score Function    ####

def mi_scoring(house_data, target):
    """Takes DataFrame containing features and a Series containing the target values to return mutual information scores """
    # label encoding for categorical variables
    for col in house_data.select_dtypes('category'):
        house_data[col], _ = house_data[col].factorize()
    # Select all discrete features
    discrete_features = house_data.dtypes == 'int64'
    # Calculate mutual information scores, store the values in a series, and sort by score from high to low
    mi_scores = mutual_info_regression(house_data, target, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name='Mutual Information Scores', index=house_data.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


####    Feature Building Functions    ####

def labelEncoder(house_data):
    """ Label encodes a dataframe's categorical features """
    for feature in house_data.select_dtypes(["category"]):
        house_data[feature] = house_data[feature].cat.codes
    return house_data

def mathTransformations(house_data):
    """ performs mathemtical transformations on several features of DataFrame """
    # Ratio: above-ground living area divided by total lot area
    house_data['LivableArea'] = house_data['GrLivArea'] / house_data['LotArea']
    
    # Ratio: (first floor area plus second floor area) divided by the total number of rooms above ground
    house_data['RoomArea'] = (house_data['FirstFlrSF'] + house_data['SecondFlrSF']) / house_data['TotRmsAbvGrd']
    
    # Product: overall quality multiplied by overall condiiton
    house_data['QualCond'] = house_data['OverallQual'] * house_data['OverallCond']
    return house_data

def interactionTransformer(house_data):
    """ performs transformations between two features that have an established interaction """
    # Interaction between building type and above-ground living area as shown in the plot
    # One hot encode building types, then multiply by the above ground living area to get interaction feature columns
    interaction_features = pd.get_dummies(house_data['BldgType'], prefix='BldgArea')
    interaction_features = interaction_features.mul(house_data['GrLivArea'], axis=0)
    
    # Interaction between building type and garage area as shown in the plot
    # One hot encode building types, then multiply by the garage area to get interaction feature columns
    interaction_features2 = pd.get_dummies(house_data['GarageType'], prefix='Garage_AreabyType')
    interaction_features2 = interaction_features2.mul(house_data['GarageArea'], axis=0)
    house_data = house_data.join([interaction_features, interaction_features2])
    return house_data

def countTransformer(house_data):
    """ Adds a new feature that describes the presence or absence of outdoor features as a sum """
    porch_types = ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ThreeSsnPorch', 'ScreenPorch']
    # Add 1 to the count if the value at that index is greater than 1
    house_data['PorchTypes'] = house_data[porch_types].gt(0).sum(axis=1)
    return house_data

def groupTransformer(house_data):
    """ Groups a feature then applies a transformation to the grouped data """
    # Group by neighborhood then fill in values with the median above-ground living area of the group
    house_data["MeadianNbhdLivableArea"] = house_data.groupby('Neighborhood')['GrLivArea'].transform('median')
    return house_data

def kMeansCluster(house_data, clustering_features, n_clusters, cluster_name):
    """ apply a k-means clustering algorithm to the dataset, given input features """
    # Standardize so that one variable is not heavily weighted in the clustering
    house_data_scaled = house_data.loc[:, clustering_features]
    house_data_scaled = (house_data_scaled - house_data_scaled.mean(axis=0)) / house_data_scaled.std(axis=0)

    # Fit the KMeans model to house_data_scaled and add the cluster labels
    kmeans = KMeans(n_clusters=n_clusters, n_init = 10, random_state=0)
    house_data[cluster_name] = kmeans.fit_predict(house_data_scaled)
    return house_data



# This is the code I intended to use for imputation, but it strips the column dtypes that were just created

# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer

# # Define a numerical and categorical imputer
# numImputer = SimpleImputer(strategy='constant', fill_value=0)
# catImputer = SimpleImputer(strategy='constant', fill_value = 'None')

# # Find the names of categorical and numerical features and store each in a list
# numerical_features   = [colname for colname in house_data_encoded if house_data_encoded[colname].dtype in ['int64','float64']]
# categorical_features = [colname for colname in house_data_encoded if house_data_encoded[colname].dtype in ['category']]

# # Combine the imputers to a Column Transformer to apply both transformations at once
# imputePreprocessor = ColumnTransformer(
#     transformers=[
#     ('numerical', numImputer, numerical_features),
#     ('categorical', catImputer, categorical_features),
# ])

# # Impute missing values and store as a new DataFrame
# house_data_imputed = pd.DataFrame(imputePreprocessor.fit_transform(house_data_encoded)) 

# # Reaaply column names as they were stripped during imputation
# house_data_imputed.columns = house_data_encoded.columns
