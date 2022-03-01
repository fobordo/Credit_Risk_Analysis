# Credit Risk Analysis

## Overview
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. The purpose of the Credit Risk Analysis was to employ different techniques to train and evaluate machine learning models with unbalanced classes. The imbalanced-learn and scikit-learn libraries were used to build and evaluate models using resampling.

Using the [credit card credit dataset](/Resources/LoanStats_2019Q1.csv) from LendingClub, a peer-to-peer lending services company, the data was oversampled using the `RandomOverSampler` and `SMOTE` methods, and undersample using the `ClusterCentroids` method. Then, a combinatorial approach of over- and under-sampling was impemented using SMOTE and Edited Nearest Neighbors (ENN) -- also known as the `SMOTEENN` method. Next, two machine learning models that reduce bias, `BalancedRandomForestClassifier` and `EasyEnsembleClassifier`, were compared to predict credit risk. Finally, the performance of these models was evaluated to see how well they predict data, and a recommendation was made on whether they should be used to predict credit risk. 

## Results
In evaluating the performance of each model, it was important to keep in mind the importance of high recall over high precision in the case of predicting credit risk, since false negatives are a bigger concern (the model predicting that someone will not default on the loan but they do) than false positives (the model predicting that someone will default but they don't).

### Oversampling with `RandomOversampler`

In the first model, the data was oversampled using the `RandomOversampler` method. The model yielded the following results.

#### Balanced Accuracy Score
![randomoversampler_bal_acc_score](/Resources/Images/randomoversampler_bal_acc_score.png)
- Balanced Accuracy Score: 0.64

#### Classification Report
![randomoversampler_classification_report](/Resources/Images/randomoversampler_classification_report.png)
- Precision Score for Predicting High Risk: 0.01
- Precision Score for Predicting Low Risk: 1.00
- Recall Score for Predicting High Risk: 0.63
- Recall Score for Predicting Low Risk: 0.64

#### Result Summary

Overall, the `RandomOversampler` model yielded unimpressive results. While the precision for predicting low risk credit was perfect, the precision for predicting high risk was extremely low, which was indicative of a large number of false positives. It is also questionable that the precision for predicting low risk credit was perfect, and further testing should be performed. Since the balanced accuracy score and recall scores for predicting both high and low risk credit were relatively low, this model would not be great at detecting credit risk.

### Oversampling with `SMOTE`

In the second model, the data was oversampled using the `SMOTE` method. The model yielded the following results.

#### Balanced Accuracy Score
![smote_bal_acc_score](/Resources/Images/smote_bal_acc_score.png)
- Balanced Accuracy Score: 0.65

#### Classification Report
![smote_classification_report](/Resources/Images/smote_classification_report.png)
- Precision Score for Predicting High Risk: 0.01
- Precision Score for Predicting Low Risk: 1.00
- Recall Score for Predicting High Risk: 0.62
- Recall Score for Predicting Low Risk: 0.68

#### Result Summary

Similar to the `RandomOversampler` model, the `SMOTE` model also yielded unimpressive results. Like the previous model, while the precision for predicting low risk credit was perfect, the precision for predicting high risk was extremely low, which was indicative of a large number of false positives. As with the previous model, it is questionable that the precision for predicting low risk credit was perfect, and further testing should be performed. In this model, we also saw a slight increase in the recall score for predicting low risk credit. However, since the balanced accuracy score and recall scores for predicting both high and low risk credit were relatively low, this model would not be great at detecting credit risk.

### Undersampling with `ClusterCentroids`

In the third model, the data was undersampled using the `ClusterCentroids` method. The model yielded the following results.

#### Balanced Accuracy Score
![clustercentroids_bal_acc_score](/Resources/Images/clustercentroids_bal_acc_score.png)
- Balanced Accuracy Score: 0.52

#### Classification Report
![clustercentroids_classification_report](/Resources/Images/clustercentroids_classification_report.png)
- Precision Score for Predicting High Risk: 0.01
- Precision Score for Predicting Low Risk: 1.00
- Recall Score for Predicting High Risk: 0.57
- Recall Score for Predicting Low Risk: 0.46

#### Result Summary

Similar to the previous two models, the `ClusterCentroids` model yielded unimpressive, and even worse results. Like the previous models, while the precision for predicting low risk credit was perfect, the precision for predicting high risk was extremely low, which was indicative of a large number of false positives. As with the previous models, it is questionable that the precision for predicting low risk credit was perfect, and further testing should be performed. Further, the recall scores for predicting both high and low risk credit were low, which is indicative of a large number of false negatives. Due to these factors, as well as a low balanced accuracy score, this model would not be great at detecting credit risk.

### Combination (Over and Under) Sampling with `SMOTEENN`

In the fourth model, the data was over- and undersampled using the `SMOTEENN` method. The model yielded the following results.

#### Balanced Accuracy Score
![smoteenn_bal_acc_score](/Resources/Images/smoteenn_bal_acc_score.png)
- Balanced Accuracy Score: 0.65

#### Classification Report
![smoteenn_classification_report](/Resources/Images/smoteenn_classification_report.png)
- Precision Score for Predicting High Risk: 0.01
- Precision Score for Predicting Low Risk: 1.00
- Recall Score for Predicting High Risk: 0.71
- Recall Score for Predicting Low Risk: 0.59

#### Result Summary

Similar to the previous three models, the `SMOTEENN` model yielded unimpressive. Like the previous models, while the precision for predicting low risk credit was perfect, the precision for predicting high risk was extremely low, which was indicative of a large number of false positives. As with the previous models, it is questionable that the precision for predicting low risk credit was perfect, and further testing should be performed. In this model, we saw an increase in the recall score for predicting high risk credit, but a low recall score for predicting low risk credit. Due to the low balanced accuracy, precision, and recall scores, this model would not be great at detecting credit risk.

### Undersampling with `BalancedRandomForestClassifier`

In the fifth model, the data was undersampled using the `BalancedRandomForestClassifier` method. The model yielded the following results.

#### Balanced Accuracy Score
![balancedrandomforestclassifier_bal_acc_score](/Resources/Images/balancedrandomforestclassifier_bal_acc_score.png)
- Balanced Accuracy Score: 0.73

#### Classification Report
![balancedrandomforestclassifier_classification_report](/Resources/Images/balancedrandomforestclassifier_classification_report.png)
- Precision Score for Predicting High Risk: 0.02
- Precision Score for Predicting Low Risk: 1.00
- Recall Score for Predicting High Risk: 0.62
- Recall Score for Predicting Low Risk: 0.84

#### Result Summary

The `BalancedRandomForestClassifier` model yielded slightly better results than the previous four models. Like the previous models, while the precision for predicting low risk credit was perfect, the precision for predicting high risk was extremely low, which was indicative of a large number of false positives. As with the previous models, it is questionable that the precision for predicting low risk credit was perfect, and further testing should be performed. In this model, we saw an increase in the recall score for predicting both high and low risk credit, and a higher balanced accuracy score. However, due to the extremely low precision score for predicting high risk credit, this model would not be great at detecting credit risk.

### Undersampling with `EasyEnsembleClassifier`

In the fifth model, the data was undersampled using the `EasyEnsembleClassifier` method. The model yielded the following results.

#### Balanced Accuracy Score
![easyensembleclassifier_bal_acc_score](/Resources/Images/easyensembleclassifier_bal_acc_score.png)
- Balanced Accuracy Score: 0.93

#### Classification Report
![easyensembleclassifier_classification_report](/Resources/Images/easyensembleclassifier_classification_report.png)
- Precision Score for Predicting High Risk: 0.09
- Precision Score for Predicting Low Risk: 1.00
- Recall Score for Predicting High Risk: 0.92
- Recall Score for Predicting Low Risk: 0.94

The `EasyEnsembleClassifier` model yielded significantly better results than the previous five models. Like the previous models, while the precision for predicting low risk credit was perfect, the precision for predicting high risk was extremely low, which was indicative of a large number of false positives. As with the previous models, it is questionable that the precision for predicting low risk credit was perfect, and further testing should be performed. In this model, we saw an increase in the recall score for predicting both high and low risk credit, and a high balanced accuracy score. However, due to the extremely low precision score for predicting high risk credit, this model would not be great at detecting credit risk.

## Summary
Out of all the machine learning models, the model that could best predict credit risk was the `EasyEnsembleClassifier` model, which yielded a high balanced accuracy score (0.93), high precision for predicting low risk (1.00), and high recall scores for predicting high risk (0.92) and low risk (0.94). However, the precision score for predicting high risk was extremely low (0.09), which was a common issue among all of the machine learning models. As such, though the `EasyEnsembleClassifier` model would perform better than the other models, I would ultimately not recommend any of these models for predicting high risk credit due to the extremely low precision scores for predicting high risk.