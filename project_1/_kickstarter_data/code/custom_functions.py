import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def custom_split(data, predict, rseed, scaling, encoding, pca, poly, features):
    X = pd.DataFrame()
    y = data[predict]
    
    std = StandardScaler()
    mm = MinMaxScaler()
    one_hot = OneHotEncoder()
    le = LabelEncoder()
    
    if "goal" in features:
        if scaling == "std":
            goal_std = pd.DataFrame(std.fit_transform(data["usd_goal_real"].to_numpy().reshape(-1, 1)))
            X = pd.concat((X, goal_std), axis=1)
        elif scaling == "minmax":
            goal_mm = pd.DataFrame(mm.fit_transform(data["usd_goal_real"].to_numpy().reshape(-1, 1)))
            X = pd.concat((X, goal_mm), axis=1)
        else:
            X = pd.concat((X, data["usd_goal_real"]), axis=1)
            
    if "time" in features:
        if scaling == "std":
            time_std = pd.DataFrame(std.fit_transform(pd.to_numeric(data["elapsed_time"]).to_numpy().reshape(-1, 1)))
            X = pd.concat((X, time_std), axis=1)
        elif scaling == "minmax":
            time_mm = pd.DataFrame(mm.fit_transform(pd.to_numeric(data["elapsed_time"]).to_numpy().reshape(-1, 1)))
            X = pd.concat((X, time_mm), axis=1)
        else:
            X = pd.concat((X, pd.to_numeric(data["elapsed_time"])), axis=1)
            
    if "category" in features:
        if encoding == "onehot":
            cat_oh = pd.DataFrame(one_hot.fit_transform(data[["category"]]).toarray())
            X = pd.concat((X, cat_oh), axis=1)
        elif encoding == "label":
            cat_le = pd.DataFrame(le.fit_transform(data["category"]), columns=["category"])
            X = pd.concat((X, cat_le), axis=1)
            
    if "main_category" in features:
        if encoding == "onehot":
            main_cat_oh = pd.DataFrame(one_hot.fit_transform(data[["main_category"]]).toarray())
            X = pd.concat((X, main_cat_oh), axis=1)
        elif encoding == "label":
            main_cat_le = pd.DataFrame(le.fit_transform(data["main_category"]), columns=["main_category"])
            X = pd.concat((X, main_cat_le), axis=1)
            
    if "country" in features:
        if encoding == "onehot":
            country_oh = pd.DataFrame(one_hot.fit_transform(data[["country"]]).toarray())
            X = pd.concat((X, country_oh), axis=1)
        elif encoding == "label":
            country_le = pd.DataFrame(le.fit_transform(data["country"]), columns=["country"])
            X = pd.concat((X, country_le), axis=1)
            
    if pca != False:
        pca_fitted = PCA(n_components=pca).fit_transform(X)
        X = pd.DataFrame(data=pca_fitted[:,:2])
        
    if poly != False:
        polynomial = PolynomialFeatures(poly)
        X = polynomial.fit_transform(np.array(X))
        y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rseed)
    
    return X_train, X_test, y_train, y_test