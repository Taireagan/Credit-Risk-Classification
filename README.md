<div align="center">
    <h1>Credit Risk Prediction</h1>
</div>

<div align="center">
    <h4>By: Tai Reagan</h4>
</div>

<div align="center">
    <img src="https://github.com/Taireagan/Credit-Risk-Classification/blob/main/Images/credit_risk.jpg" alt="credit_risk" width="500"/>
</div>

---
<a name="top"></a>
# Table of Contents

<details>
  <summary>Click to expand</summary>


- [Purpose of the Analysis](#purpose-of-the-analysis)
- [Resources](#resources)
- [Stages of the Machine Learning Process](#stages-of-the-machine-learning-process)
  - [Financial Information](#financial-information)
  - [Basic Understanding of Machine Learning ](#basic-understanding-of-machine-learning)
  - [Variables to Predict](#variables-to-predict)
  - [Creating Training and Testing Data](#creating-training-and-testing-data)
  - [Logistic Regression Model](#logistic-regression-model)
  - [Confusion Maxtrix](#confusion-maxtrix)
  - [Classification Report Results](#classification-report-results)
- [Summary](#summary)
- [Recommendation](#recommendation)



</details>



## Purpose of the Analysis
The purpose of this analysis is to predict the credit risk of borrowers by applying machine learning techniques to assess their likelihood of defaulting on a loan. Through the evaluation of a dataset containing historical lending activity, the objective is to develop a model that can determine a borrower's creditworthiness based on several financial characteristics.

## Resources 
The data used for the analysis was provided by [Lending Data](https://github.com/Taireagan/Credit-Risk-Classification/blob/main/Resources/lending_data.csv)




## Stages of the Machine Learning Process

### Financial Information

The dataset provided was from a peer-to-peer lending service company and contained critical financial details, including loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, derogatory marks, total debt, and loan status. The goal was to predict the loan status, which indicated whether the borrower had either defaulted (represented by 0) or successfully repaid the loan (represented by 1).

### Basic Understanding of Machine Learning 

Before proceeding, the diagram below illustrates the fundamental process of training and testing machine learning models. The full dataset is first organized into two components: features (input variables) and the target (the outcome to be predicted), which will be elaborated on later. The dataset is then randomly split into two subsets: the training set, which is used to train the model, and the test set, which is used to evaluate its performance. After the model is trained on the training set, it is tested on the test set to assess its ability to accurately predict new data.

<div align="center">
    <img src="https://github.com/Taireagan/Credit-Risk-Classification/blob/main/Images/4_train-test-split.jpg" alt="example" width="600"/>
</div>

[Back to Top](#top)

### Variables To Predict

The first step in preparing the data for machine learning is to load in your original data and view columns provided.
<div align="center">
    <img src="https://github.com/Taireagan/Credit-Risk-Classification/blob/main/Images/original%20data.png" alt="orginal_data" width="600"/>
</div>


To continue the analysis, the next step involves creating the label set and feature set. The label set, referred to as y, contains the target variable, which in this case is the loan status. In machine learning, the label is the outcome that the model is trying to predict. For this analysis, the loan status will be the prediction output, where a value of 0 represents a loan that has defaulted, and a value of 1 indicates a loan that has been repaid.

The feature set, referred to as X, is created by removing the loan status column from the dataset. The remaining columns, which include variables such as loan size, interest rate, and borrower income, will serve as the input variables for the machine learning model. These input variables, or features, help determine whether a loan will default or be repaid.

The machine learning algorithm will then use the feature set (X) to predict the values of the target variable (y), allowing it to evaluate the likelihood of a borrower defaulting on a loan.

<div align="center">
    <img src="https://github.com/Taireagan/Credit-Risk-Classification/blob/main/Images/spliting%20data.png" alt="splitting_data" width="600"/>
</div>


[Back to Top](#top)



### Creating Training and Testing Data


This code is used in the process of preparing data for a machine learning model. It splits the data into two parts: a training set and a test set. The training set is used to teach the machine learning model how to make predictions based on the data, while the test set is used to evaluate how well the model performs on new, unseen data.

By dividing the data in this way, the model can be trained and tested independently, which helps ensure that the model's predictions will work well on real-world data. The function train_test_split from the sklearn library is used to do this splitting, and the random_state=1 ensures that the data is split in the same way every time the code is run, which makes the results reproducible.

<div align="center">
    <img src="https://github.com/Taireagan/Credit-Risk-Classification/blob/main/Images/training%20and%20testing.png" alt="training_and_testing" width="600"/>
</div>

[Back to Top](#top)



### Logistic Regression Model

This diagram shows the basic concept of logistic regression, which is a machine learning algorithm used for binary classification (where the output can be one of two possible values, like 0 or 1).

In logistic regression, the algorithm predicts the likelihood that something will happen, such as whether a loan will be repaid or defaulted. The predicted values lie between 0 and 1, and they are shown on the S-shaped curve in the image. If the prediction is closer to 0, the outcome is more likely to be negative (e.g., loan default), and if it's closer to 1, the outcome is more likely to be positive (e.g., loan repaid).

The graph shows this process visually: on the X-axis are the input values (features), and the Y-axis shows the predicted probability. The logistic regression model produces an output that follows the S-shaped curve, mapping input features to probabilities between 0 and 1.

<div align="center">
    <img src="https://github.com/Taireagan/Credit-Risk-Classification/blob/main/Images/Logistic%20regression.png" alt="logistic regression" width="600"/>
</div>



This code is part of the process of building and training a logistic regression model. Here's how it works:

1. Importing the Logistic Regression model: The first line imports the logistic regression function from the sklearn library, which is a popular tool for machine learning in Python.

2. Creating the model: The second line creates an instance of the logistic regression model, where random_state=1 is set to ensure that the model produces consistent results every time the code is run.

3. Training the model: The final line uses the .fit() function to train the model using the training data (X_train and y_train). In this step, the model learns the relationship between the input features (X) and the target variable (y) by analyzing the patterns in the data. Once trained, the model can make predictions about whether a borrower is likely to default or repay a loan based on new input data.

This process helps the model understand how to make predictions for unseen data in the future.

<div align="center">
    <img src="https://github.com/Taireagan/Credit-Risk-Classification/blob/main/Images/LR%20model.png" alt="lr_model" width="600"/>
</div>


[Back to Top](#top)


### Confusion Maxtrix


The diagram below represents a confusion matrix, which is used to evaluate the performance of a classification model in machine learning. It compares the actual outcomes (what really happened) with the predicted outcomes (what the model predicted).

- **True Positive (TP):** The model correctly predicted a positive outcome (e.g., correctly identified a loan that was repaid).
- **False Positive (FP):** The model incorrectly predicted a positive outcome when it was actually negative (e.g., predicted a loan would be repaid, but it defaulted).
- **True Negative (TN):** The model correctly predicted a negative outcome (e.g., correctly identified a loan that defaulted).
- **False Negative (FN):** The model incorrectly predicted a negative outcome when it was actually positive (e.g., predicted a loan would default, but it was repaid).

The confusion matrix helps in understanding how well the model is performing by showing where it is making correct or incorrect predictions.

<div align="center">
    <img src="https://github.com/Taireagan/Credit-Risk-Classification/blob/main/Images/confusion%20matrix.png" alt="confusion_matrix" width="600"/>
</div>

This code generates a confusion matrix to assess the performance of a machine learning model in predicting loan outcomes. The process begins by utilizing the confusion_matrix() function, which takes the actual outcomes (y_test) and the model’s predicted outcomes (testing_predictions) to create the matrix. The next step involves converting this matrix into a DataFrame using pd.DataFrame(). The DataFrame is structured with clearly labeled rows and columns, representing the actual loan outcomes ("Actual Healthy Loan" and "Actual High-Risk Loan") and the predicted outcomes ("Predicted Healthy Loan" and "Predicted High-Risk Loan"). This process ultimately produces the output displayed below the code.

<div align="center">
    <img src="https://github.com/Taireagan/Credit-Risk-Classification/blob/main/Images/confusion%20matrix%20output.png" alt="confusion_matrix_output" width="600"/>
</div>

The matris shows:
- **18655** actual healthy loans were correctly predicted as healthy loans (true positives).
- **110** actual healthy loans were incorrectly predicted as high-risk loans (false positives).
- **36** actual high-risk loans were incorrectly predicted as healthy loans (false negatives).
- **583** actual high-risk loans were correctly predicted as high-risk loans (true negatives).

This output allows for evaluating the model’s accuracy by comparing how well the model predicted healthy versus high-risk loans. The numbers help assess both correct and incorrect predictions made by the model.

[Back to Top](#top)


### Classification Report Results

In this code, a classification report and accuracy score are generated to evaluate the performance of a machine learning model. The process is broken down into two parts:

1. Classification Report: 
    - The classification_report() function is used to produce a detailed summary of the model’s performance. It takes the actual outcomes (y_test) and the predicted outcomes (testing_predictions) as inputs.
    -  The report includes important metrics such as precision, recall, f1-score, and support for each class (in this case, class "0" and class "1").

2. Accuracy Score:
    - The accuracy_score() function calculates the overall accuracy of the model, which is simply the percentage of correct predictions out of the total predictions. This value is printed at the bottom.

<div align="center">
    <img src="https://github.com/Taireagan/Credit-Risk-Classification/blob/main/Images/classification%20report.png" alt="classification_report" width="600"/>
</div>

Here is a breakdown of the report:
- **Precision:** Measures how many of the predicted positives (e.g., predicted repaid loans) were actually correct. For class "0" (healthy loans), the precision is 1.00, meaning all healthy loans predicted were correct.
- **Recall:** Measures how many of the actual positives were correctly predicted. For class "0", a recall of 0.99 means that 99% of actual healthy loans were correctly predicted.
- **F1-score:** This is a combined measure of precision and recall, providing a balance between the two. A higher score indicates better model performance.
- **Support:** Shows how many instances of each class were in the actual data. In this case, there were 18,765 healthy loans (class "0") and 619 high-risk loans (class "1").
- **Accuracy:** The overall accuracy of the model is 0.99 (or 99.2%), which means that the model made correct predictions for 99.2% of all test cases.
- **Macro avg** and **Weighted avg:** These averages summarize the overall performance across both classes. The macro average gives equal weight to both classes, while the weighted average takes into account the number of instances in each class.

Overall, the report indicates that the model performed exceptionally well, with high precision, recall, and accuracy across both loan categories.

[Back to Top](#top)

### Summary

In conclusion the model performs exceptionally well in predicting **Healthy Loans (0)** with perfect precision and almost perfect recall. However, it is slightly less effective in predicting **High-Risk Loans (1)**. While the precision for high-risk loans is **84%**, the recall is higher at **94%**, meaning the model is more likely to identify actual high-risk loans but occasionally misclassifies healthy loans as high-risk. The logistic regression model is very accurate overall, especially for healthy loans, but there is a slight trade-off in precision for high-risk loan predictions. Overall the model correctly identifies both healthy and high-risk loans, though its performance is slightly better at predicting healthy loans.

[Back to Top](#top)

### Recommendation
If predicting high-risk loans (class "1") is the most important aspect of the problem (to avoid defaults), it may be worth optimizing for recall in class "1" by adjusting the model threshold or trying other models that might boost performance for this class. However, if overall performance is the priority, with a slight emphasis on correctly identifying healthy loans, the current logistic regression model with 99.2% accuracy and a balanced F1-score across classes would be recommended.

[Back to Top](#top)





