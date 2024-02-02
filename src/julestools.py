import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.utils import resample

def nan_to_null(df, cols=None):
    if cols is None: cols = df.columns.tolist()
    [df[k].fillna(0, inplace=True) for k in cols ]


def eval_model(note, model, X_test, y_test, results, label_type=1):
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    precision = precision_score(y_test,y_pred, pos_label=label_type)
    recall = recall_score(y_test, y_pred, pos_label='Yes')
    f1 = f1_score(y_test, y_pred, pos_label='Yes')
    false_negatives = confusion_matrix(y_test, y_pred)[1][0]
    new_result = pd.DataFrame({'note':note,'accuracy':score,'precision':precision,'recall':recall,'f1_score':f1,'false_negatives':false_negatives},index=[0])
    return pd.concat([results,new_result],axis=0)
