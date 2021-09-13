# Diabetes-Prediction

#### Goal: To develop a POC using Flask, HTML and CSS for predicting whether a person is suffering from Diabetes or not, implementing Machine Learning algorithm.

### About the Data set:
This is a machine learning project where we will predict whether a person is suffering Diabetes or not. 
The dataset was downloaded from Kaggle. The datasets consists of eight medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.
The target column says whether the person is having the disease or not based on the predictor variables.
The target has two values 1(having the diease) and 0(not having the disease). A binary classification problem statement.

### Project Description:
After loading the dataset("diabetes.csv") the first step was to perform an extensive Exploratory Data Analysis(EDA). The EDA part comprises of creating countplots for the target feature to check whether the dataset is balanced or not. It was a balanced dataset. Density plots were made to check the distribution of each features.
Similarly, histograms were made for the same purpose. Boxplots were created for outliers detection. Some amount of outliers present in few features. Then, the dataset was divided into independent(x) features and Dependent(Y) features for the purpose of Data Analysis. A correlation heatmap was made to check the correlation between all the independent features. Again, a correlation heatmap was plotted, but this time the target feature was also taken into consideration.
Scatter plots were made to visualize the direction of correlations.

The second step was to perform Feature Engneering. The initial step was to check for null values. Then the zero values of every feature was calculated. The zero values of every feature was replaced by mean using Sklearn's SimpleImputer.

The third step was Feature Selection. As the dataset had only 8 independent features, features were selected manually based on domain knowledge. 'Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction' and 'Age'were the features that got selected. Feature Scaling was not required as XGBoost does that internally.

The Forth step was Model Building. The dataset was divided into independent(X) and dependent(y) features. Train test split was performed for getting the train and test datasets.
XGBoost classifier was applied on the training data after testing with other Machine Learning algorithmns. Predicton and validaion was performed on the test dataset.

The fifth step was to perform Hyperparameter Optimization on our model. A range of parameters for "learning_rate", "max_depth", "min_child_weight", "gamma" and "colsample_bytree" was selected and passed through RandomizedSearchCV. The model was then fitted with the best parameters. The main aim was to reduce the False Positives and the False Negatives. Model performed really good and validated based on classification report, confusion matrix and accuracy score.

The final step was to save the model as a pickle file to reuse it again for the Deployment purpose. Joblib was used to dump the model at the desired location.

Deployment Architecture: The model was deployed locally (port 5000). The backend part of the application was made using Flask and for the frotend part HTML and CSS was used.
I have not focussed much on the frontend as I am not that good at it. The file "app.py" contains the entire flask code and inside the templates folder, "diabetes.html" contains the homepage and "result.html" contains the result page. 
