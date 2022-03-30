# This file includes some useful functions throughout the course.

# Libraries
import re 
import numpy as np
import pandas as pd
import scipy.stats as ss


# 3. Data Loading
def to_snake_case(names):
    """
    Converts list of camel case and double underscore to snake case.
    Modified from:
    https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
    """

    def converter(name):
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        name = re.sub('__([A-Z])', r'_\1', name)
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
        return name.lower()
    return [converter(name) for name in names]

def show_columns_name(df, width=3):
    """
    Returns a list of the column names of the dataframe.
    """
    total_items = len(df.columns)
    cnames = df.columns.tolist()
    rows = [cnames[x:x+width] for x in range(0, total_items, width)]
    output = pd.DataFrame(rows).fillna('')
    return output
# 4. EDA
def df_information(df_input):
    df = df_input.copy()

    output = pd.DataFrame()
    for cname, cvalue in df.items():
        dtype = ""
        unique = cvalue.nunique()
        if unique == 1:
            dtype="no variance"
        elif unique==2:
            dtype="boolean"
        elif unique==len(cvalue):
            dtype="identifier"
        elif pd.api.types.is_numeric_dtype(cvalue):
            dtype="numerical"
        else:
            dtype="categorical"

        row = {
                'name': cname,
                'type': dtype,
                'dtype': str(df[cname].dtype),
                'unique': unique,
                'missing': cvalue.isnull().sum(),
                'count': len(cvalue)
            }


        output = output.append(row, ignore_index=True)
    
    return output

def correlation_matrix(dfi, cats, nums):
    """Return correlation matrix of all types.
    Categorical VS Categorical: Cramers V
    Categorical VS Numerical: Correlation Ratio
    Numerical VS Numerical: Pearson's Correlation Coefficient
    Parameters:
        - dfi: dataframe
        - cats: list of categorical columns names
        - nums: list of numerical columns names"""
    def cramers_v(x, y):
        confusion_matrix = pd.crosstab(x,y)
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2/n
        r,k = confusion_matrix.shape
        phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
        rcorr = r-((r-1)**2)/(n-1)
        kcorr = k-((k-1)**2)/(n-1)
        return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

    def correlation_ratio(categories, measurements):
        fcat, _ = pd.factorize(categories)
        cat_num = np.max(fcat)+1
        y_avg_array = np.zeros(cat_num)
        n_array = np.zeros(cat_num)
        for i in range(0,cat_num):
            cat_measures = measurements[np.argwhere(fcat == i).flatten()]
            n_array[i] = len(cat_measures)
            y_avg_array[i] = np.average(cat_measures)
        y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
        numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
        denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
        if numerator == 0:
            eta = 0.0
        else:
            eta = numerator/denominator
        return eta  

    def corr(baris, kolom):
        if baris == kolom:
            return 1
        # do categorical vs categorical
        if baris in cats and kolom in cats:
            return cramers_v(df[baris], df[kolom])
        # do numerical vs numerical
        if baris in nums and kolom in nums:
            return df[baris].corr(df[kolom])
        # do categorical vs numerical
        if baris in cats and kolom in nums:
            return correlation_ratio(df[baris], df[kolom])
        # do numerical vs categorical
        if baris in nums and kolom in cats:
            return correlation_ratio(df[kolom], df[baris])
        print(baris, kolom)
        return -5
    df = dfi.copy()
    cnames = nums + cats

    output = pd.DataFrame()

    output['cnames'] = cnames
    output.set_index('cnames', inplace=True)
    # i = kolom


    for kolom in cnames:
        values = []
        for baris in cnames:
            value = corr(baris, kolom)
            values = values + [value]
        
        output[kolom] = values
    return output

# import standard scaler from sklearn
from sklearn.preprocessing import StandardScaler
# import ordinal encoder from sklearn
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
#import PCA
from sklearn.decomposition import PCA
def visualize2D(df, target_name):
    scaler = StandardScaler()
    encoder = OrdinalEncoder()
    vis = df.drop([target_name], axis=1).copy()
    numerical_cols = [cname for cname in vis.columns if vis[cname].dtype in ['int64', 'float64']]
    categorical_cols = [cname for cname in vis.columns if vis[cname].dtype == 'object']
    if len(numerical_cols) > 0:
        vis[numerical_cols] = scaler.fit_transform(vis[numerical_cols])

    if len(categorical_cols) > 0:
        vis[categorical_cols] = encoder.fit_transform(vis[categorical_cols])

    colors = encoder.fit_transform(df['churn'].values.reshape(-1, 1))
    pca = PCA(n_components=2)
    vis = pca.fit_transform(vis)
    # return vis
    plt.scatter(vis[:, 0], vis[:, 1], c=colors, cmap='coolwarm')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('2D PCA')
    plt.show()


# 5. Data Preprocessing
# import train test split
from sklearn.model_selection import train_test_split

def train_val_test_split(X, y, val_size=0.2, test_size=0.2,  random_state=231019):
    """Split dataframe into train, validation, and test sets.
    Parameters:
        - X: X dataframe
        - y: y dataframe
        - target_name: name of target column
        - test_size: size of test set
        - val_size: size of validation set
    Returns:
        - X_train, X_val, X_test, y_train, y_val, y_test dataframes"""
    X = X.copy()
    y = y.copy()
    # split dataframe into train, val, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # split train set into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/(1-test_size)*val_size, random_state=random_state)
    # return train, val, and test sets
    return X_train, X_val, X_test, y_train, y_val, y_test

# 6. Model Definition

# 7. Model Training

# 8. Model Evaluation

# 9. Model Saving

# 10. Model Inference

# 11. Conclusion
