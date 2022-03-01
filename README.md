# Credit Risk Analysis

## Overview
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. The purpose of the Credit Risk Analysis was to employ different techniques to train and evaluate machine learning models with unbalanced classes. The imbalanced-learn and scikit-learn libraries were used to build and evaluate models using resampling.

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, the data was oversampled using the `RandomOverSampler` and `SMOTE` methods, and undersample using the `ClusterCentroids` method. Then, a combinatorial approach of over- and under-sampling was impemented using SMOTE and Edited Nearest Neighbors (ENN) -- also known as the `SMOTEENN` method. Next, two machine learning models that reduce bias, `BalancedRandomForestClassifier` and `EasyEnsembleClassifier`, were compared to predict credit risk. Finally, the performance of these models was evaluated to see how well they predict data, and a recommendation was made on whether they should be used to predict credit risk.

## Results
Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all six machine learning models. Use screenshots of your outputs to support your results.

### Oversampling with `RandomOversampler`

In the first model, the data was oversampled using the `RandomOversampler` method. The model yielded the following results.

#### Balanced Accuracy Score
![randomoversampler_bal_acc_score](/Resources/Images/randomoversampler_bal_acc_score.png)
- Balanced Accuracy Score: 

#### Classification Report
![randomoversampler_classification_report](/Resources/Images/randomoversampler_classification_report.png)
- Precision Score for High Risk: 
- Recall Score:

### Oversampling with `SMOTE`

Balanced Accuracy Score:
![smote_bal_acc_score](/Resources/Images/smote_bal_acc_score.png)

Classification Report:
![smote_classification_report](/Resources/Images/smote_classification_report.png)
- Precision Score:
- Recall Score:

### Undersampling with `ClusterCentroids`

Balanced Accuracy Score:
![clustercentroids_bal_acc_score](/Resources/Images/clustercentroids_bal_acc_score.png)

Classification Report:
![clustercentroids_classification_report](/Resources/Images/clustercentroids_classification_report.png)
- Precision Score:
- Recall Score:

### Combination (Over and Under) Sampling with `SMOTEENN`

Balanced Accuracy Score:
![smoteenn_bal_acc_score](/Resources/Images/smoteenn_bal_acc_score.png)

Classification Report:
![smoteenn_classification_report](/Resources/Images/smoteenn_classification_report.png)
- Precision Score:
- Recall Score:

### Undersampling with `BalancedRandomForestClassifier`

Balanced Accuracy Score:
![balancedrandomforestclassifier_bal_acc_score](/Resources/Images/balancedrandomforestclassifier_bal_acc_score.png)

Classification Report:
![balancedrandomforestclassifier_classification_report](/Resources/Images/balancedrandomforestclassifier_classification_report.png)
- Precision Score:
- Recall Score:

### Undersampling with `EasyEnsembleClassifier`

Balanced Accuracy Score:
![easyensembleclassifier_bal_acc_score](/Resources/Images/easyensembleclassifier_bal_acc_score.png)

Classification Report:
![easyensembleclassifier_classification_report](/Resources/Images/easyensembleclassifier_classification_report.png)
- Precision Score:
- Recall Score:

## Summary
Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.