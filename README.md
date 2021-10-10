# Diabetes-Prediction

#### Goal: To develop a POC using Flask, HTML and CSS for predicting whether a person is suffering from Diabetes or not, implementing Machine Learning algorithm.

### About the Data set:
This is a machine learning project where we will predict whether a person is suffering from Diabetes or not. The dataset was downloaded from [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database). The datasets consists of eight medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on. The target column says whether the person is having the disease or not based on the predictor variables. The target has two values 1(having the diease) and 0(not having the disease). A binary classification problem statement.

### Project Description:

After loading the dataset("diabetes.csv") the first step was to perform an extensive Exploratory Data Analysis(EDA). The EDA part comprises of creating countplots for the target feature to check whether the dataset is balanced or not. It was a balanced dataset. Density plots were made to check the distribution of each features. Similarly, histograms were made for the same purpose. Boxplots were created for outliers detection. Some amount of outliers present in few features. Then, the dataset was divided into independent(x) features and Dependent(Y) features for the purpose of Data Analysis. A correlation heatmap was made to check the correlation between all the independent features(x). Again, a correlation heatmap was plotted, but this time the target feature was also taken into consideration. Scatter plots were made to visualize the distribution of the data points and decide the type of correlation.

The second step was to perform Feature Engneering. The initial step was to check for null values. Then the zero values of every feature was calculated. The zero values of every feature was replaced by mean using Sklearn's SimpleImputer.

The third step was Feature Selection. As the dataset had only 8 independent features, features were selected manually based on domain knowledge. 'Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction' and 'Age'were the features that got selected. Feature Scaling was not required as XGBoost does that internally.

The Forth step was Model Building. The dataset was divided into independent(X) and dependent(y) features. Train test split was performed for getting the train and test datasets.
XGBoost classifier was applied on the training data after testing with other Machine Learning algorithmns. Predicton and validaion was performed on the test dataset.

The fifth step was to perform Hyperparameter Optimization on our model. A range of parameters for "learning_rate", "max_depth", "min_child_weight", "gamma" and "colsample_bytree" was selected and passed through RandomizedSearchCV. The model was then fitted with the best parameters. The main aim was to reduce the False Positives and the False Negatives. Model performed really good and validated based on classification report, confusion matrix and accuracy score.

The final step was to save the model as a pickle file to reuse it again for the Deployment purpose. Joblib was used to dump the model at the desired location.

The "Diabetes Prediction.ipynb" file contains all these informations.

### Deployment Architecture: 
The model was deployed locally (port: 5000). The backend part of the application was made using Flask and for the frotend part HTML and CSS was used. I have not focussed much on the frontend as I am not that good at it. The file "app.py" contains the entire flask code and inside the templates folder, "diabetes.html" contains the homepage and "result.html" contains the result page. 

### Installation:
The Code is written in Python 3.7.3 If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:

##### 1. First create a virtual environment by using this command:
```bash
conda create -n myenv python=3.7
```
##### 2. Activate the environment using the below command:
```bash
conda activate myenv
```
##### 3. Then install all the packages by using the following command
```bash
pip install -r requirements.txt
```
##### 4. Then, in cmd or Anaconda prompt write the following code:
```bash
python app.py
```
##### Make sure to change the directory to the root folder.  

### A Glimpse of the application:
![Screenshot (153)](https://user-images.githubusercontent.com/75041273/133078617-311d8e64-7a47-4fec-be2a-71458b55e2fc.png)
![Screenshot (154)](https://user-images.githubusercontent.com/75041273/133078572-e4047598-dd10-4f63-a179-23fdbd98ff00.png)
![Screenshot (152)](https://user-images.githubusercontent.com/75041273/133078717-499126ef-88c1-40b6-9f40-5011831e74ff.png)



### Further Changes to be Done:
- [ ] Including the remaining two features, that might increase model accuracy.
- [ ] Deploying the Web Application on Cloud.
     - [ ] Google Cloud 
     - [ ] Azure
     - [ ] Heroku
     - [ ] AWS

### Technology Stack:

<img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen" /> <img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" /> <img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white" /> ![Seaborn](https://img.shields.io/badge/Seaborn-%230C55A5.svg?style=for-the-badge&logo=seaborn&logoColor=%white)  <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" /> <img src="https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white" /> <img src="https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white"/> <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white" />  <img src="https://img.shields.io/badge/matplotlib-342B029.svg?&style=for-the-badge&logo=matplotlib&logoColor=white"/> <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" /> <img src="https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon" />

### Diabetes Prediction using Pytorch:
The above project is also executed using Pytorch ANN and trained on the entire dataset containing all the features. Model worked well with accuracy score of 79.2% with less false positives and negatives and good precision and recall.
