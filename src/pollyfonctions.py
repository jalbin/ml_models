

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
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn import linear_model
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, RFE
from sklearn.tree import DecisionTreeClassifier


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
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, cross_validate
    from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
    from sklearn import linear_model
    from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, RFE
    from sklearn.tree import DecisionTreeClassifier




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
        print(f'Empty "{name}":\n{empty}\n')

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
    

def spliting(X, y, t_size):
       '''take 2 dataframes X and y and split it unto train and test
       return
       X_train, X_test, y_train, y_test '''
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size, random_state=7986)
       return X_train, X_test, y_train, y_test
# X_train, X_test, y_train, y_test = split(X, y, t_size)

def scaling(data):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
    return data_scaled

def scaler_X_Y(X_train, X_test, y_train, y_test, X_data):
    ''' X_train, X_test, y_train, y_test  scaler_data(X_train, X_test, y_train, y_test)'''
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    X_train = pd.DataFrame(X_train, columns=X_data.columns)
    X_test = pd.DataFrame(X_test, columns=X_data.columns)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    return X_train, X_test, y_train, y_test

# X_train, X_test, y_train, y_test = scaler_data(X_train, X_test, y_train, y_test)



def print_roc_curve(model_name, model, X_test, y_test):
    Y_score = model.predict_proba(X_test)[:, 1]
    # Check for non-finite values in Y_score
    if not np.all(np.isfinite(Y_score)):
        # Handle non-finite values by replacing them with a large finite value
        Y_score[np.isnan(Y_score)] = np.max(Y_score[np.isfinite(Y_score)])
    fpr, tpr, thresholds = roc_curve(y_test, Y_score)
    plt.title(f"ROC curve for {model_name}")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.plot(fpr, tpr)



def split_scaler_data(X, y, t_size):
    print(X.head())
    print(y.head())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= t_size, random_state=7986)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    return X_train, X_test, y_train, y_test

# X_train, X_test, y_train, y_test = split_scaler_data(X, y, t_size)

def fit_predict_evaluate_fonction(model_name, model, X_train, X_test, y_train, y_test, results):
    ''' fit predicts and evaluate model and add results to a table results'''
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    score = model.score(X_test, y_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    ''' Recall'''
    #(y_pred & y_test).sum() / y_test.sum()
    print(f"Model score: {score}")
    print(f"F1 score: {f1}")
    print(f"accuracy score: {accuracy}")
    print(f"precision_score: {precision}")
    print(f"recall_score: {recall}")
    confusion_matr = confusion_matrix(y_test, y_pred)
    print(f"y_pred & y_test: {(y_pred & y_test).sum()}")
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
    disp = print_roc_curve(model_name, model, X_test, y_test)
    if disp is not None:
        disp.plot(ax=ax2)
    print(results)
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
def model_errors(y_test, y_pred):
    '''print a histogram with the model errors'''
    errors = (y_test - y_pred).abs() / y_test
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
        




def eliminate_highly_correlated_columns_based_test(df, df_kbest, high_corr_threshold=0.8):
    ''' this fonction takes a dataframe df, a dataframe of k_best scores, a high_corr_threshold= 
    and returns a new dataframe without the columns highly correlated, and a list of columns to drop'''
    # Calculate correlation matrix
    correlations_matrix = df.corr().abs()
    
    # Get the intersection of columns between df and df_kbest
    common_columns = list(set(df.columns) & set(df_kbest['column_name']))

    # Initialize a set to store columns to eliminate
    columns_to_eliminate = set()

    for i in range(len(correlations_matrix.columns)):
        for j in range(i):
            col_i = correlations_matrix.columns[i]
            col_j = correlations_matrix.columns[j]
            
            # Check if both columns are present in the common_columns list
            if col_i in common_columns and col_j in common_columns:
                # Check if correlation is above the threshold
                if abs(correlations_matrix.loc[col_i, col_j]) > high_corr_threshold:
                    # Choose which column to eliminate based on k-best scores
                    score_i = df_kbest.loc[df_kbest['column_name'] == col_i, 'kbest_score'].values[0]
                    score_j = df_kbest.loc[df_kbest['column_name'] == col_j, 'kbest_score'].values[0]

                    # Keep the one with lower k-best score
                    column_to_eliminate = col_i if score_i < score_j else col_j
                    columns_to_eliminate.add(column_to_eliminate)
    df_result = df.drop(columns=columns_to_eliminate)
    print("Original DataFrame Shape:")
    print(df.shape)
    print("\nThis is the list of highly correlated columns to eliminate:")
    print(columns_to_eliminate)
    print("\nDataFrame after eliminating highly correlated columns:")
    print(df_result.shape)
    # Create a new DataFrame without the eliminated columns
   

    return df_result, columns_to_eliminate



# generating data frames




def generates_X_best_ktest(df, columns_ranks, number_features):
    ''' Takes a dataframe, a columns rank dataframe, a number_features and generates the X dataframe'''
    columns_top_kbest = list(columns_ranks.sort_values(by = ['kbest_score'], ascending = False).head(number_features)['column_name'])
    X_data = df[columns_top_kbest]
    return X_data

# X = generates_X_df_best_ktest(df, columns_ranks, 20)


def generate_X_top20_rfe(df, columns_ranks):
    ''' Takes a dataframe, a columns rank dataframe and 
    generates the X dataframe with the top rfe_rank columns'''
    columns = list(columns_ranks[columns_ranks["rfe_rank"] == 1].sort_values(["kbest_score"], ascending=False)["column_name"])
    X_data = df[columns]
    return X_data

# X = generate_X_df_best_rfe(df, columns_ranks)



def generate_X_best_rfe(df, columns_ranks, features):
    ''' Takes a dataframe, a columns rank dataframe, a number_features and generates the X dataframe'''
    columns = list(columns_ranks[columns_ranks["rfe_rank"] <= features].sort_values(["kbest_score"], ascending=False)["column_name"])
    X_data = df[columns]
    return X_data

# X = generate_X_df_best_rfe(df, columns_ranks)

def choice_rank(rank, features, df, columns_ranks):
    '''takes a rank between rfe and kbest and generates X from a dataframe'''
    if rank == "rfe":
        X_data = generate_X_best_rfe(df, columns_ranks, features)
    if rank == "kbest":
         X_data = generate_X_best_rfe(df, columns_ranks, features)
    return X_data


    
