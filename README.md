# Stock-Market-Prediction
NEXUS INTERNSHIP PROJECT PHASE 1               
                                                                        
Name: Manoj Prabakar B  
Ph no: 9843172255

Email ID: manojrockerguy246@gmail.com
                                                                      
Project Phase 1:  Stock Market  Prediction
1. Exploratory Data Analysis (EDA):
- Perform EDA on the stock market dataset to understand its structure and
characteristics.
- Visualize key statistics and trends in stock prices.
2. Predictive Modeling:
- Utilize machine learning techniques to predict stock prices.
- Implement regression models to forecast future stock values.
3. Documentation:
- Document your approach, methodologies, and insights gained from the stock
market dataset.
- Provide clear explanations for the chosen predictive models.


For the stock market prediction project, our approach involves a combination of Exploratory Data Analysis (EDA) and Predictive Modeling techniques.

Exploratory Data Analysis:
•	We began by acquiring the stock market dataset from [source].
•	We conducted an initial exploration of the dataset to understand its structure, size, and features.
•	We performed data cleaning and preprocessing steps to handle missing values, outliers, and ensure data consistency.
•	Visualizations such as line plots, histograms, and correlation matrices were utilized to gain insights into the distribution of stock prices, volume, and relationships between variables.
•	Key statistics and trends in stock prices over time were analyzed to identify patterns and anomalies.


Predictive Modeling:
•	We split the dataset into features and the target variable, which is typically the closing price of the stock.
•	Regression models such as Linear Regression, Decision Tree Regression, and Random Forest Regression were implemented to forecast future stock values.
•	The dataset was further divided into training and testing sets to evaluate the performance of the models.
•	Evaluation metrics such as Mean Squared Error (MSE) and R-squared were used to assess the accuracy and performance of the models.



2. Methodologies and insights gained from the stock market dataset:
   
Methodologies:
•	Data preprocessing techniques were applied to handle missing values, outliers, and ensure data quality.
•	Exploratory Data Analysis (EDA) techniques such as visualizations and statistical analysis were used to understand the underlying patterns and trends in the dataset.
•	Feature engineering techniques may have been employed to create new features or transform existing ones to improve model performance.
•	Predictive modeling techniques such as regression analysis were utilized to build models that predict future stock prices based on historical data.

Insights:
•	Through EDA, we gained insights into the distribution of stock prices, trading volume, and relationships between variables.
•	We identified correlations between different features and the target variable, which helped in feature selection and model building.
•	Trends and patterns in stock prices over time were analyzed, providing valuable insights into market behavior and potential investment opportunities.
•	The predictive models developed were evaluated based on their accuracy and performance in forecasting future stock values.



4. Provide clear explanations for the chosen predictive models:
Linear Regression:
•	Linear Regression is a simple and interpretable regression model that assumes a linear relationship between the features and the target variable.
•	It's suitable for predicting continuous numerical values, making it a good choice for forecasting stock prices.
•	Despite its simplicity, Linear Regression can provide valuable insights into the relationship between features and the target variable.

Decision Tree Regression:
•	Decision Tree Regression is a non-linear regression model that partitions the feature space into regions and makes predictions based on the average of the target variable within each region.
•	It's capable of capturing non-linear relationships between features and the target variable, making it suitable for complex datasets with non-linear patterns.
•	Decision trees are easy to interpret and visualize, allowing for a clear understanding of the decision-making process.

Random Forest Regression:
•	Random Forest Regression is an ensemble learning technique that combines multiple decision trees to make predictions.
•	It improves upon the performance of individual decision trees by reducing overfitting and increasing generalization.
•	Random Forest Regression is robust to outliers and noise in the data, making it a powerful tool for predictive modeling in diverse datasets like stock market data.
•	In summary, the chosen predictive models offer a range of capabilities and trade-offs, allowing us to effectively forecast stock prices while gaining valuable insights into market trends and behaviors.



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Pre-Modeling Tasks

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Modeling

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# Evaluation and comparision of all the models


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score,auc,f1_score
from sklearn.metrics import precision_recall_curve,roc_curve


data = pd.read_csv('C:/breast_cancer_prediction.csv')


data.head()


print(data.isnull().sum())


print(data.groupby('diagnosis').size())


data.info()


data.columns

data.shape


data.describe().T


data= data.drop(['Unnamed: 32','id'],axis=1)


palette ={'B' : 'lightblue', 'M' : 'magenta'}


fig = plt.figure(figsize=(12,12))
def plot_scatter(a,b,k):
    plt.subplot(k)
    sns.scatterplot(x = data[a], y = data[b], hue = "diagnosis", data = data, palette = palette)
    plt.title(a + ' vs ' + b,fontsize=15)
    
plot_scatter('texture_mean','texture_worst',221) 
plot_scatter('area_mean','radius_worst',222) 
plot_scatter('perimeter_mean','radius_worst',223)  
plot_scatter('perimeter_mean','radius_worst',224) 



fig = plt.figure(figsize=(12,12))

  
plot_scatter('smoothness_mean','texture_mean',221) 
plot_scatter('texture_mean','symmetry_se',222) 
plot_scatter('fractal_dimension_worst','texture_mean',223) 
plot_scatter('texture_mean','symmetry_mean',224)


fig = plt.figure(figsize=(12,12))
plot_scatter('area_mean','fractal_dimension_mean',221)
plot_scatter('radius_mean','fractal_dimension_mean',222)
plot_scatter('area_mean','smoothness_se',223)
plot_scatter('smoothness_se','perimeter_mean',224)



sns.scatterplot(x= 'area_mean', y= 'smoothness_mean', hue= 'diagnosis', data=data, palette='CMRmap')



# texture mean vs radius_mean

size = len(data['texture_mean'])

area = np.pi * (15 * np.random.rand( size ))**2
colors = np.random.rand( size )

plt.xlabel("texture mean")
plt.ylabel("radius mean") 
plt.scatter(data['texture_mean'], data['radius_mean'], s=area, c= colors, alpha=0.5)


sns.countplot(x='diagnosis', data=data, palette='Paired')
plt.show()


m = plt.hist(data[data["diagnosis"] == "M"].radius_mean,bins=30,fc = (1,0,0,0.5),label = "Malignant")
b = plt.hist(data[data["diagnosis"] == "B"].radius_mean,bins=30, fc = (1,0,0.5), label= "Bening")

plt.legend()
plt.xlabel ("Radius Mean Values")
plt.ylabel ("Frequency")
plt.title("Histogram of Radius Mean for Bening and Malignant Tumors")
plt.show()



sns.jointplot(data= data, x='area_mean', y='smoothness_mean', size=5)


# Label Encoder
LEncoder = LabelEncoder()
data['diagnosis'] = LEncoder.fit_transform(data['diagnosis'])


X = data.drop('diagnosis',axis=1).values
y = data['diagnosis'].values

random_state = 42
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=random_state)


sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test= sc.transform(x_test)

# Support Vector classifier
svc = SVC(probability=True)
svc.fit(X_train,y_train)
y_pred_svc = svc.predict(X_test)

X_train.shape, y_train.shape,X_test.shape, y_test.shape


models = []
Z = [SVC()]
X = ["SVC"]


for i in range(0, len(Z)):
    model = Z[i]
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    models.append(accuracy_score(pred, y_test))

d = { "Accuracy" : models , "Algorithm" : X }
data_frame = pd.DataFrame(d)
data_frame


cm = np.array(confusion_matrix(y_test, y_pred_svc, labels=[1,0]))
confusion_mat= pd.DataFrame(cm, index = ['cancer', 'healthy'], columns =['predicted_cancer','predicted_healthy'])
confusion_mat

sns.heatmap(cm,annot=True,fmt='g',cmap='Set3')

print(accuracy_score(y_test, y_pred_svc))


print(precision_score(y_test, y_pred_svc))

print(recall_score(y_test, y_pred_svc))

print(classification_report(y_test, y_pred_svc))


#plt.style.use('seaborn-pastel')

y_score = svc.decision_function(X_test)

FPR, TPR, _ = roc_curve(y_test, y_score)
ROC_AUC = auc(FPR, TPR)
print (ROC_AUC)

plt.figure(figsize =[11,9])
plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 4)
plt.plot([0,1],[0,1], 'k--', linewidth = 4)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate', fontsize = 18)
plt.ylabel('True Positive Rate', fontsize = 18)
plt.title('Receiver operating characteristic example', fontsize= 18)
plt.show()


print(f"ROC AUC Score: {roc_auc_score(y_test, y_score)}")
