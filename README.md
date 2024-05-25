<h1>Evaluation of the performance of machine learning models for predicting bankruptcies in Taiwan</h1>
(Évaluation de la performance de modèles de machine learning pour la prédiction de faillites à Taïwan)


## Introduction

The goal of this project was to implement and evaluate supervised learning models using the Taiwan Bankruptcy Data. We created models for the prediction/classification and compared the metrics between the different models to determine the ones that performed better. 

Models implemented: 
- Logistic regression
- NuSVC
- BernoulliNB 
- AdaBoostClassifier
- Linear Discriminant Analysis

<a href="https://docs.google.com/presentation/d/1tdpORTf7XnyaB9SCRnna4Ib4ey8XRpHnNzF5j7Y6mhs/edit?usp=sharing"> Final Presentation </a>

## Steps

1. Descriptive analytics, data cleaning and formatting
   
2.Data preparation for machine learning and Feature selection:
   
- verification of correlation and 
- reducing the number of variables, 
- feature selection using VarianceThreshold , XGBoost, Kbest and Recursive Feature 
- oversampling
- data scaling
- splitting the dataset (train/test)


3. Model research.

- Logistic regression
- NuSVC
- BernoulliNB 
- AdaBoostClassifier
- Linear Discriminant Analysis



4. Implemented the models on our data

5. Hyperparameters tuning

6. Comparing the results using metrics:
   
- accuracy
- recall
- precision
- ROC_AUC score
- plot ROC_AUC curve

