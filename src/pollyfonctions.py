

import numpy as np
import pandas as pd
import seaborn as sns
from plotly import express as px
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import plot_tree, export_text
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.decomposition import PCA
import re # regex
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def import_models():
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from plotly import express as px
    from matplotlib import pyplot as plt
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.tree import plot_tree, export_text
    from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, f1_score
    from sklearn.decomposition import PCA
    import re # regex
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn import svm
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix




def print_exploratory_data(data):
    ''' data = {"dataname": dataframe}'''
    '''EDA: it takes a dictionary with one or more dataframes and prints the information about it'''
    print("Data shape: \n")
    for name, d in data.items():
        num_rows, num_columns = d.shape
        print(f'Shape of "{name}" : {num_columns} columns, {num_rows} rows')
    print("\n Duplicates:\n")
    for name, d in data.items():
        num_duplicates = d.duplicated().sum()
        print(f'Number of duplicates in "{name}" : {num_duplicates} duplicates')
    print("\n")
    for name, d in data.items():    
        unique_v = d.nunique()
        print(f'Unique Values "{name}":\n{unique_v}\n')
    for name, d in data.items():    
        typ = d.dtypes
        print(f'Data Type "{name}":\n{typ}\n')
    for name, d in data.items():    
        empty = d.isna().sum()
        print(f'Empty "{name}":\n{typ}\n')

    for name, d in data.items():
        num_columns = d.select_dtypes(np.number).columns
        str_columns = d.select_dtypes(object).columns
        print(f'Numerical Columns "{name}": \n{num_columns}\n ')
        print(f'Non numerical Columns "{name}": \n{str_columns}\n ')

def map_columns(data,columns_map):
    '''data[columns_map] = map_columns(data,columns_map)'''
    data[columns_map] = data[columns_map].replace(mapping).fillna(data[columns_map])
    return data[columns_map]
# data[columns_map] = map_columns(data,columns_map)      
# data = data.apply(lambda x: 1 if x == "5" else 0) 

def print_max_min_values(dataframe):
    '''print_overall_max_min_values(dataframe)'''
    overall_max_value = dataframe.values.max()
    overall_min_value = dataframe.values.min()
    print(f"Overall Max Value: {overall_max_value}")
    print(f"Overall Min Value: {overall_min_value}")

# print_overall_max_min_values(dataframe)
    

def spliting(X, y):
       '''take 2 dataframes X and y and split it unto train and test
       return
       X_train, X_test, y_train, y_test = split(X, y)'''
       print(X.head())
       print(y.head())
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=7986)
       return X_train, X_test, y_train, y_test
# X_train, X_test, y_train, y_test = split(X, y)



def scaler_data(X_train, X_test, y_train, y_test, X_data):
    ''' X_train, X_test, y_train, y_test = scaler_data(X_train, X_test, y_train, y_test)'''
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    X_train = pd.DataFrame(X_train, columns=X_data.columns)
    X_test = pd.DataFrame(X_test, columns=X_data.columns)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    return X_train, X_test, y_train, y_test

# X_train, X_test, y_train, y_test = scaler_data(X_train, X_test, y_train, y_test)



def print_y_score(model_name, model, X_test, y_test):
    Y_score = model.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, Y_score)
    plt.title(f"ROC curve for {model_name}")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.plot(fpr, tpr);


def fit_predict_evaluate_fonction(model_name, model, X_train, X_test, y_train, y_test, results):
    ''' fit predicts and evalueate model and add results to a table results'''
    results = pd.DataFrame(columns=['model_name','accuracy','precision','recall','f1_score', 'false_negatives'])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    score = model.score(X_test, y_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    ''' Recall'''
    #(y_pred & y_test).sum() / y_test.sum()
    print(f"score: {score}")
    print(f"accuracy score: {accuracy}")
    print(f"precision_score: {precision}")
    print(f"recall_score: {recall}")
    confusion_matr = confusion_matrix(y_test, y_pred)
    print((y_pred & y_test).sum())
    ''' Create subplots'''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    disp = ConfusionMatrixDisplay(confusion_matr, display_labels=model.classes_)
    disp.plot(ax=ax1)
    ax1.set_title('Confusion Matrix')
    
    '''model_evaluation:'''
    f1 = f1_score(y_test, y_pred)
    false_negatives = confusion_matr[1][0]
    new_result = pd.DataFrame({'model_name': model_name,'accuracy': accuracy,'precision':precision,'recall':recall,'f1_score':f1,'false_negatives':false_negatives},index=[0])   
    results = pd.concat([results, new_result],axis=0)
    print_y_score(model_name, X_test, y_test)
    return accuracy, precision, recall, y_pred, results


# model_name = "Logistic regression"
# model = LogisticRegression()
# accuracy, precision, recall, y_pred_lr, results = fit_predict_evaluate_fonction(model_name, model, X_train, X_test, y_train, y_test, results)
# results
# print_y_score(model_name, X_test, y_test)


def data_dummy(data): # handling 
    data_copy = data.copy()
    data_dummy = pd.get_dummies(data_copy, drop_first=True)
    print(data_dummy.dtypes)
    print(f"columns: \n{data_dummy.columns}")
    return data_dummy

#data = data_dummy(data)
#data

# .
def corr_matrix(data):
    '''will create a correlation matrix using the numeric columns in the dataset'''
    '''correlations_matrix = corr_matrix(websites)'''
    correlations_matrix = data.select_dtypes(np.number).corr() # store our correlation matrix
    return correlations_matrix

# correlations_matrix = corr_matrix(websites)
# correlations_matrix

# high_corr_threshold = 0.8
# highly_correlated_pairs = []

def corre_cols(correlations_matrix, high_corr_threshold, highly_correlated_pairs):
    '''still doesn`t work
    it takes a correlations matrix and create a list of highly correlated columns in a dataframe
    highly_correlated_pairs = list_correlated_col(correlations_matrix, high_corr_threshold)'''
    #highly_correlated_pairs = []
    for i in range(len(correlations_matrix.columns)):
        for j in range(i):
            if abs(correlations_matrix.iloc[i, j]) > high_corr_threshold:
                col_pair = (correlations_matrix.columns[i], correlations_matrix.columns[j], correlations_matrix.iloc[i, j])
                highly_correlated_pairs.append(col_pair)

    #print(f"Number of highly correlated pairs: {len(highly_correlated_pairs)}")
    #return highly_correlated_pairs

# highly_correlated_pairs = list_correlated_col(correlations_matrix, high_corr_threshold)


def print_highly_correlated_pairs(highly_correlated_pairs):
    ''' it takes a list of correlated columns and print it'''
    for pair in highly_correlated_pairs:
        print(f"Columns: {pair[0]} and {pair[1]} - Correlation: {pair[2]}")


def create_heatmap(data, dataname):
    '''2 Create a heatmap using `seaborn` to visualize which columns have high collinearity.'''
    correlations_matrix = corr_matrix(data)
    mask = np.triu(np.ones_like(correlations_matrix, dtype=bool))
    cmap = sns.diverging_palette(135, 135, s=90, l=40, center="light", as_cmap=True)
    plt.figure(figsize=(12, 10))  # Adjust the width and height as needed
    ax = sns.heatmap(correlations_matrix, annot=True, cmap=cmap, center=0, mask=mask)
    ax.set_title(f'Correlation Matrix {dataname}')
    plt.show()

# create_heatmap(data)
def drop_col(data, coltodrop):   
    ''' it takes a data and a list of columns to drop'''
    '''#data = drop_col(data, coltodrop)'''
    '''# data = data.drop(columns=['NUMBER_SPECIAL_CHARACTERS','APP_PACKETS'])'''
    data = data.drop(columns= coltodrop)
    return data

#data = drop_col(data, coltodrop)

def fix_col_names(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(r'\s+','_',regex=True)
    return df


#models
def model_errors(y_test, y_test_pred):
    '''print a histogram with the model errors'''
    errors = (y_test - y_test_pred).abs() / y_test
    errors.hist()
    print(f"Model errors: {errors}")
    return errors

def print_classes(dataseries):
    '''print value counts from a dataframe series'''
    count_classes = dataseries.value_counts()
    print(count_classes)
    count_classes.plot(kind = 'bar')



from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


def evaluate_model(model, X, y, cv, scoring_metrics):
    results = {}

    for metric in scoring_metrics:
        if metric == 'precision_score':
            # Calculate precision scores
            precision_scores = []
            for train_idx, test_idx in cv.split(X, y):
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                y_pred = model.predict(X.iloc[test_idx])
                precision = precision_score(y.iloc[test_idx], y_pred)
                precision_scores.append(precision)
            mean_precision = sum(precision_scores) / len(precision_scores)
            results['precision_score'] = mean_precision
            print(f"Precision Score: {mean_precision:.4f}")
        else:
            # Use cross_val_score for other metrics
            scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
            mean_score = scores.mean()
            results[metric] = mean_score
            print(f"{metric}: {mean_score:.4f}")

    return results

# List of models
# list_models = [model_dt, model_knn, model_lr]
# scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']
# Specify the number of folds
# num_folds = 5
# Create a stratified K-fold cross-validator
# Specify the scoring metrics you want to evaluate
# Loop through each model and evaluate 
def evaluate_models(list_models, scoring_metrics, num_folds, X_train, y_train): 
    '''# Loop through each model and evaluate it'''
    ''' it takes a list of models, a list of scoring_metrics ('accuracy', 'precision', 'recall', 'f1'), num_folds ''' 
    cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    for model in list_models:
        print(f"Evaluating Model: {type(model).__name__}")
        results = evaluate_model(model, X_train, y_train, cv, scoring_metrics)
        print(results)
        print("\n")
    return results
        


