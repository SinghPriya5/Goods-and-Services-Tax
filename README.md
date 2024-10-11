<p align="center">
<img  width='400' height='300' src="https://github.com/SinghPriya5/Goods-and-Services-Tax/blob/main/static/images/GST_GIF.gif"></p>

<div style="text-align: center; color: #2c3e50; font-family: 'Trebuchet MS', sans-serif;">
  <h1 style='color:#e74c3c; font-size: 3em; letter-spacing: 2px;'>✿ 𝓖𝓸𝓸𝓭𝓼 𝓢𝓮𝓻𝓿𝓲𝓬𝓮𝓼 𝓪𝓷𝓭 𝓣𝓪𝔁 𝓒𝓵𝓪𝓼𝓼𝓲𝓯𝓲𝓬𝓪𝓽𝓲𝓸𝓷 ✿</h1>
</div>
<img align="right" width="500" height="550" src="https://github.com/SinghPriya5/Goods-and-Services-Tax/blob/main/static/images/TAX.jfif">

<h3>📜 Table of Content</h3>

* [Problem Statement](#Problem-Statement)
* [Types](#Types)
* [Goal](#Goal)
* [Approach](#Approach)
* [Data Collection](#Data-Collection)
* [Data Preparation and Cleaning](#Data-Preparation-and-Cleaning)
* [Model Development](#Model-Development)
* [Model Evaluation](#Model-Evaluation)
* [Model Performance](#Model-Performance)
* [Tools Used](#Tools-Used)
* [Model Accuracy](#Model-Accuracy)
* [Continuous Improvement](#Continuous-Improvement)
* [Deployed](#Deployed)
* [Model Interpretation](#Model-Interpretation)
* [Web View](#Web-View)
* [Bug or Feature Request](#Bug-or-Feature-Request)
* [Future Scope](#Future-Scope)
* [Project Overview](#Project-Overview)
* [Conclusion](#Conclusion)

## <h3>📜Problem Statement:</h3>
<ul style="font-family: 'Courier New', monospace;">
  <h2>Problem Statements for GST Classification</h2>

<ol>
    <li>
        <h3>Classify Transactions</h3>
        <p><strong>Objective:</strong> Classify transactions into different GST categories based on features.</p>
        <p><strong>Problem:</strong> How accurately can we classify transactions based on historical data?</p>
    </li>
    <li>
        <h3>Identify Key Features</h3>
        <p><strong>Objective:</strong> Determine which features are most influential in classification.</p>
        <p><strong>Problem:</strong> What features contribute most to accurate classification of GST categories?</p>
    </li>
    <li>
        <h3>Analyze Model Performance</h3>
        <p><strong>Objective:</strong> Compare performance of various models in classifying GST.</p>
        <p><strong>Problem:</strong> Which model provides the best accuracy and why?</p>
    </li>
    <li>
        <h3>Class Imbalance</h3>
        <p><strong>Challenge:</strong> The main challenge in this project is the severe class imbalance in the dataset. Approximately 91% of the data is labeled as 0 (negative class), while only 9% is labeled as 1 (positive class). This imbalance makes it difficult for machine learning models to identify and predict the minority class (1) accurately because most models tend to favor the majority class (0).</p>
    </li>
</ol>

## <h3>📜Types:</h3>

<p><img width="700" height="500" src="https://github.com/SinghPriya5/Goods-and-Services-Tax/blob/main/static/images/TYPE.png"></p>

## <h3>📜Goal:</h3>
<div style="font-family: 'Courier New', monospace; font-size: 1.1em;">
  The main goals of the GST classification project are:
  <ul>
    <li><b>Improve Accuracy:</b> Enhance the accuracy of GST classification models.</li>
    <li><b>Feature Importance:</b> Identify important features affecting classification.</li>
    <li><b>Provide Insights:</b> Deliver actionable insights from the classification results.</li>
  </ul>
</div>

## <h3>📜Approach:</h3>
<div style="font-family: 'Courier New', monospace; font-size: 1.1em;">
  The analysis involves data preprocessing, feature selection, model training, and evaluation. We will apply different algorithms and analyze their performance.
</div>

## <h3>📜Data Collection:</h3>
1. **Loading Datasets:** Two CSV files containing training data were loaded into Pandas DataFrames. One dataset contained input features, and the other contained the target labels.
   ```python
   import pandas as pd

   train1 = pd.read_csv('Train_Data_Input.csv')
   train2 = pd.read_csv('Train_Data_Target.csv')
2. **Merging Datasets:** The two DataFrames were merged on a common identifier (ID) to create a single comprehensive dataset that includes both features and target labels.
   ```python
   merged_df = pd.merge(train1,train2, on='ID')
   
## <h3>📜Data Preparation and Cleaning:</h3>
**Data Cleaning:**
 * **Removing Duplicates:** Any duplicate records in the dataset were removed to ensure data integrity.
   ```python
     merged_df.drop_duplicates(inplace=True)

* **Handling Missing Values:** Missing values in specific columns were filled with the median of those columns. This method helps maintain the distribution of the data without introducing bias.
   ```python
    cols_to_fill = ['Column0', 'Column3', 'Column4', 'Column5', 'Column6', 'Column8', 'Column9', 'Column10', 'Column14', 'Column15']
    merged_df[cols_to_fill] = df[cols_to_fill].fillna(df[cols_to_fill].median())
* **Outlier Detection and Removal:**
Z-Score Calculation: A z-score was calculated for a specified column (Column1) to identify outliers. The z-score indicates how many standard deviations a data point is from the mean.
  ```python
  from scipy import stats

  z_scores = stats.zscore(merged_df['Column1'])
  merged_df = merged_df[(z_scores > -2) & (z_scores < 3)]

* **Outlier Removal:** Records with z-scores outside the range of -2 to +3 were excluded from the dataset to ensure that extreme values do not skew the model training.
* **Final Dataset Preparation:**
* **Combining Cleaned Data:** The cleaned dataset (without outliers) was merged back with the original dataset to create a final dataset for analysis.
* **Feature and Target Split:** The final dataset was split into features (X) and target labels (y), where the target label indicates the classification outcome.
  ```python

  X = merged_df.drop('target', axis=1)
  y = merged_df['target']
* **Data Splitting:**
  **1.)Train-Test Split:** The dataset was divided into training and testing sets using an 80-20 split. This division allows the model to learn from one portion of the data while being evaluated on a separate portion, reducing the risk of overfitting.
  ```python
  from sklearn.model_selection import train_test_split

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## <h3>📜Model Development:</h3>
* **Logistic Regression:** This model was trained to predict the target class based on the input features. It's suitable for binary classification problems and provides interpretability through coefficients.
  ```python
  from sklearn.linear_model import LogisticRegression
 
  lr_model = LogisticRegression()
  lr_model.fit(X_train, y_train)

* **Random Forest Classifier:** An ensemble method that builds multiple decision trees and merges their predictions for more robust results. It's particularly effective for handling non-linear data.
  ```python
  from sklearn.ensemble import RandomForestClassifier

  rf_model = RandomForestClassifier()
  rf_model.fit(X_train, y_train)
* **Decision Tree with AdaBoost:** A decision tree model was combined with the AdaBoost algorithm, which adjusts the weights of the training samples based on their classification errors, improving the model's performance.
  ```python
  from sklearn.ensemble import AdaBoostClassifier
  from sklearn.tree import DecisionTreeClassifier

  dt_model = DecisionTreeClassifier()
  ada_model = AdaBoostClassifier(base_estimator=dt_model)
  ada_model.fit(X_train, y_train)
* **XGBoost:** A powerful gradient boosting technique that optimizes computation and handles overfitting well. It was trained with default hyperparameters suitable for initial evaluations.
  ```python
  from xgboost import XGBClassifier

  xgb_model = XGBClassifier()
  xgb_model.fit(X_train, y_train)
  
## <h3>📜Model Evaluation:</h3>
* **Performance Metrics:** After training each model, various performance metrics were computed:
* **Classification Report:** Provides precision, recall, F1-score, and support for each class, offering insights into the model's performance across different categories.
* **Confusion Matrix:** A matrix that visualizes the performance of the classification model, showing true positives, false positives, true negatives, and false negatives.
* **Accuracy Score:** The overall accuracy of the model, calculated as the ratio of correctly predicted instances to the total instances.
  
## <h3>📜Model Performance:</h3>

  from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

* **XGBoost**
   ```python
   y_pred_xgb = xgb_model.predict(X_test)
   print("=== XGBoost ===")
   print("Classification Report:")
   print(classification_report(y_test, y_pred_xgb))
   print("Confusion Matrix:")
   print(confusion_matrix(y_test, y_pred_xgb))
   print("Recall for class '1':", recall_score(y_test, y_pred_xgb))

* **Random Forest**
   ```python
   y_pred_rf = rf_model.predict(X_test)
   print("\n=== Random Forest ===")
   print("Classification Report:")
   print(classification_report(y_test, y_pred_rf))
   print("Confusion Matrix:")
   print(confusion_matrix(y_test, y_pred_rf))
   print("Recall for class '1':", recall_score(y_test, y_pred_rf))

* **AdaBoost**
  ```python
  y_pred_ada = ada_model.predict(X_test)
  print("\n=== AdaBoost ===")
  print("Classification Report:")
  print(classification_report(y_test, y_pred_ada))
  print("Confusion Matrix:")
  print(confusion_matrix(y_test, y_pred_ada))
  print("Recall for class '1':", recall_score(y_test, y_pred_ada))
* **Best Parameters**
* XGBoost: {'subsample': 0.8, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.2}
* Random Forest: {'n_estimators': 100, 'min_samples_split': 4, 'min_samples_leaf': 2, 'max_depth': 13, 'bootstrap': True}
* AdaBoost: {'n_estimators': 50, 'learning_rate': 0.01}

* **Final Model:**
  ```python
   model = XGBClassifier({'subsample': 0.8, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.2})
* **Model Dump:**
 As per selected trained model is dumped to joblib format for app development.

## <h3>📜Tools Used:</h3>
<ul style="font-family: 'Courier New', monospace; font-size: 1.1em;">
  <li>Jupyter Notebook</li>
  <li>VS Code</li>
  <li>PyCharm</li>
</ul>

## <h3>📜Model Accuracy</h3>
The model achieved an accuracy of 98%.

## <h3>📜Continuous Improvement:</h3>
<div style="font-family: 'Courier New', monospace; font-size: 1.1em;"> <ul> <li>Explore additional machine learning algorithms to improve prediction accuracy.</li> <li>Implement advanced feature engineering to capture more relevant data aspects.</li> <li>Optimize the Flask application for faster response times and better scalability.</li> <li>Integrate real-time data for dynamic analysis and predictions.</li> <li>Enhance the frontend user interface for a more intuitive experience.</li> </ul> </div>

## <h3>📜Deployed:</h3>
Deployed on Render -- [Link](https://github.com/SinghPriya5/Goods-and-Services-Tax/issues)

<br> The instructions are given on [Render Documentation](https://docs.render.com/deploy-flask) to deploy a web app.

<b>Model Deployment:</b> Deploy the model as a REST API using Flask. Hosted on Render for public access.

## <h3>📜Model Interpretation:</h3>
Analyzed and interpreted the model’s predictions to ensure meaningful and accurate results.

## <h3>📜Web View:</h3>

**Frontend**

<p align="center">
  <img src="https://github.com/SinghPriya5/Goods-and-Services-Tax/blob/main/static/images/Frontend.png" alt="Frontend" width="700" height="600">
</p>

**Inserting Value and Predicted Value**

<p align="center">
  
  <img src="https://github.com/SinghPriya5/Goods-and-Services-Tax/blob/main/static/images/Inserting_value.png" alt="Inserting Value" width="700" height="500">


  
  <img src="https://github.com/SinghPriya5/Goods-and-Services-Tax/blob/main/static/images/Predicting%20value.png" alt="Predicted Value"  width="700" height="500">
</p>

## <h3>📜Bug or Feature Request:</h3>

* If you find a bug (the website couldn't handle the query and/or gave undesired results), kindly open an [issue](https://github.com/SinghPriya5/Goods-and-Services-Tax/issues) here by including your search query and the expected result.

* Users can report bugs or request new features through the following channels:

- **Email:** [Email-Id](singhpriya91636@gmail.com)
- **GitHub Repository:** [GitHub link](https://github.com/SinghPriya5/Goods-and-Services-Tax)

---

## <h3>📜Future Scope:</h3>
The project has potential for expansion in several areas:

- **Advanced Modeling:** 
  - Incorporate sophisticated models such as Neural Networks to improve classification accuracy.
  
- **Additional Datasets:** 
  - Explore and integrate more diverse datasets to enhance training and generalization.

- **Feedback Mechanism:** 
  - Implement feedback loops for continuous learning, allowing the model to improve over time based on user interactions.

- **Real-Time Analytics:**
  - Develop real-time analytics features for users to track GST classifications and trends.

- **User Training:**
  - Provide training resources for users to better understand how to use the model effectively.
  - 
 ## <h3>📜Project Overview:</h3>
 
  - **Project:** [Link](https://github.com/SinghPriya5/Goods-and-Services-Tax/blob/main/Notebook/GST_Model_Production.ipynb)

## <h3>📜Conclusion:</h3>
<div style="font-family: 'Courier New', monospace; font-size: 1.1em;"> This project on GST classification utilizes machine learning to provide valuable insights into transaction classification. By analyzing the data thoroughly, developing and evaluating multiple models, we can improve prediction accuracy and enhance the decision-making process in GST classification. <br><br> The key methodology involved a systematic approach to data preparation, cleaning, and model training. By using multiple classification techniques and robust evaluation metrics, the model development process aimed to identify the most effective method for predicting the target variable, ensuring that the resulting model is both accurate and interpretable. Future steps could involve hyperparameter tuning for optimization and cross-validation to enhance model reliability. </div> 

## 📜 Thank You 😊
Thank you for taking the time to read this document. Your interest and feedback are invaluable in improving the GST classification project. If you have any questions, suggestions, or contributions, please feel free to reach out!
