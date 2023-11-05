#!/usr/bin/env python
# coding: utf-8

# ### Heart Disease Predictions Using Supervised Learning
# 

# In[4]:


# Import necessary Libraries

# For data analysis
import pandas as pd
import numpy as np

# For data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Data pre-processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#Classifier Libraries
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Ipip install xgboost
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings("ignore")


# In[5]:


# Load the dataset
df = pd.read_csv(r"C:\Users\ADMIN\Desktop\New folder (2)\10Alytics Data Science\Machine Learning\WMD2\heart.csv")
df.head()


# ## Features in the dataset and meaning:
# 
# - age – age in years
# - sex – (1=male, 0=female)
# - cp – chest pain type (1: typical angina, 2: atypical angina, 3: non-angina pain, 4: asymptomatic)
# - treslbps – resting blood pressure (in mm Hg on admission to the hospital)
# - chol – serum cholesterol in mg/dl,
# - fbs – (fasting blood sugar>120mg/dl) (1=true, 0=false)
# - restecg – resting electrocardiographic results
# - thalach – maximum heart rate achieved
# - exang – exercise induced by angina (1=yes, 0=no)
# - oldpeak – ST depression induced by exercise relative to rest
# - slope – the slope of the peak exercise ST segment
# - ca – number of major vessels (0-3) colored by flourosopy
# - thal – 3 = normal, 6 = fixed detect, 7 = reversable detect
# - target – have disease or not (1=yes, 0=no)

# In[6]:


# For better understanding and flow of analysis, I will rename some of the columns

df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved', 'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']
df.head()


# In[7]:


# Data verification - Data type, number of features and rows, missing data, e.t.c

df.info()


# In[8]:


# Statistical Analysis of the data

df.describe()


# In[9]:


# Check for missing values

print(df.isnull().sum())


# In[10]:


# Visualization the missing data
plt.figure(figsize = (10,3))
sns.heatmap(df.isnull(), cbar=True, cmap="Blues_r")


# ### Observation
# 
# - There is no missing value in the dataset

# # Exploratory Data Analysis 
# # Univariate Analysis

# In[11]:


df.columns


# In[12]:


# Check for outliers
sns.boxplot (x=df["thalassemia"])


# In[13]:


# Check for outliers
sns.boxplot (x=df["cholesterol"])


# In[14]:


#check for outliers
sns.boxplot (x=df["resting_blood_pressure"])


# In[15]:


# check for outliers
sns.boxplot (x=df["max_heart_rate_achieved"])


# In[16]:


# Data visualization
# Age bracket

def age_bracket(age):
    if age <= 35:
        return "Youth(<=35)"
    elif age <= 55:
        return "Adult(<=55)"
    elif age <= 65:
        return "Old Adult(<=65)"
    else:
        return "Elderly(>65)"
    
df['age_bracket'] = df['age'].apply(age_bracket)

# Investigating the age group of patients

plt.figure(figsize = (10, 5))
sns.countplot(x='age_bracket', data=df)
plt.xlabel('Age Group')
plt.ylabel('count of Age Group')
plt.title('Total Number of Patients')


# ### Observation
# Based on the chart above, majority of patients age is less than or equal to 55 years.

# In[17]:


# Data visualization
# Sex

def gender(sex):
    if sex == 1:
        return "Male"
    else:
        return "Female"
    
df['gender'] = df['sex'].apply(gender)

# Investigating the age group of patients

plt.figure(figsize = (10, 5))
sns.countplot(x='gender', data=df)
plt.xlabel('Gender')
plt.ylabel('count of Patient Gender')
plt.title('Total Number of Patients')


# ### Observation
# Based on the gender, the number of male patients is more than double of the female patients. 

# In[18]:


# Data visualization
# Chest pain type (1: typical angina, 2: atypical angina, 3: non-angina pain, 4: asymptomatic)

def chest_pain(cp):
    if cp == 1:
        return "typical angina"
    elif cp == 2:
        return "atypical angina"
    elif cp == 3:
        return "non-typical angina"
    else:
        return "asymptomatic"
    
df['cp_cat'] = df['chest_pain_type'].apply(chest_pain)

# Investigating the age group of patients

plt.figure(figsize = (10, 5))
sns.countplot(x='cp_cat', data=df)
plt.xlabel('Type of Chest Pain')
plt.ylabel('count of Patient Chest Pain')
plt.title('Total Number of Patients')


# In[19]:


# Data visualization
# target - have disease or not (1=yes, 0=no)

def label(tg):
    if tg == 1:
        return "yes"
    else:
        return "no"
    
df['label'] = df['target'].apply(label)

# total patients in each category

print(df["label"].value_counts())

# Investigating the target of patients

plt.figure(figsize = (10, 5))
sns.countplot(x='label', data=df)
plt.xlabel('Target')
plt.ylabel('count of Patient Target')
plt.title('Total Number of Patients')


# # Exploratory Data Analysis
# # BIVARIATE ANALYSIS

# In[20]:


# Investigating the age group of patients by the target feature

plt.figure(figsize = (10, 5))
sns.countplot(x='age_bracket', data=df, hue='label')
plt.xlabel('Age Group')
plt.ylabel('Count of Age Group')
plt.title('Total Number of Patients')


# In[21]:


# Investigating the gender of patients by the target feature

plt.figure(figsize = (10, 5))
sns.countplot(x='gender', data=df, hue='label')
plt.xlabel('Gender')
plt.ylabel('Count of Gender')
plt.title('Total Number of Patients')


# In[22]:


# Investigating the chest pain type by the target featur

plt.figure(figsize = (10, 5))
sns.countplot(x='cp_cat', data=df, hue='label')
plt.xlabel('Types of Chest Pain')
plt.ylabel('Count of Patient Chest Pain')
plt.title('Total Number of Patients')


# # Exploratory Data Analysis 
# # Multivariate Analysis

# In[23]:


# Correlation between heart disease and other variables in the dataset
plt.figure(figsize = (10, 10))
hm = sns.heatmap(df.corr(), cbar=True, annot=True, square=True, fmt=' .2f', annot_kws={'size': 10})


# ### Observation
# Based on the heatmap presented above. There is negative and postive relationship.

# # Feature Engineering/Data Pre-Processing

# In[24]:


# Create a copy of the data (Exlude target/label alongside other columns that was created)
df1 = df[['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg',
       'max_heart_rate_achieved', 'exercise_induced_angina', 'st_depression','st_slope', 'num_major_vessels', 'thalassemia']]

label = df[['target']]


# In[25]:


df1.head()


# In[26]:


df1.dtypes


# In[27]:


# Dealing with outliers - 'resting_blood_pressure', cholesterol, thalassmia

# Normalize the data

scaler = MinMaxScaler()

df1["Scaled_RBP"] = scaler.fit_transform(df1['resting_blood_pressure'].values.reshape(-1, 1))
df1["Scaled_chol"] = scaler.fit_transform(df1['cholesterol'].values.reshape(-1, 1))
df1["Scaled_thal"] = scaler.fit_transform(df1['thalassemia'].values.reshape(-1, 1))
df1["Scaled_max_heart_rate"] = scaler.fit_transform(df1['max_heart_rate_achieved'].values.reshape(-1, 1))

df1.drop(['resting_blood_pressure', 'cholesterol', 'thalassemia', 'max_heart_rate_achieved'], axis=1, inplace=True)

df1.head()


# In[ ]:





# ## Machine Learning

# In[28]:


# Split the dataset into training and testing sets – x = questions while y = answers

X_train, X_test, y_train, y_test = train_test_split(df1, label, test_size=0.2, random_state=42)


# In[50]:


X_test.head(3)


# In[49]:


y_test.head(3)


# In[48]:


X_train.head(3)


# In[47]:


y_train.head(3)


# In[31]:


# Model Building
# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

ly_pred = logreg.predict(X_test)

print("Logistic Regression")
print("Accuracy:", accuracy_score(y_test, ly_pred))
print("Precision:", precision_score(y_test, ly_pred))
print("Recall:", recall_score(y_test, ly_pred))
print("F1-score:", f1_score(y_test, ly_pred))
print("AUC-ROC:", roc_auc_score(y_test, ly_pred))


# In[32]:


ly_pred


# In[33]:


y_test


# In[34]:


# Create a confusion matrix

lcm = confusion_matrix(y_test, ly_pred)

# Visualize the confusion matrix

sns.heatmap(lcm, annot=True, cmap="Blues", fmt="g")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[ ]:





# In[35]:


# Model Building

# Random Forest Classifier

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfy_pred = rfc.predict(X_test)
print("Logistic Regression")
print("Accuracy:", accuracy_score(y_test, rfy_pred))
print("Precision:", precision_score(y_test, rfy_pred))
print("Recall:", recall_score(y_test, rfy_pred))
print("F1-score:", f1_score(y_test, rfy_pred))
print("AUC-ROC:", roc_auc_score(y_test, rfy_pred))


# In[ ]:





# In[36]:


# Create a confusion matrix

rcm = confusion_matrix(y_test, rfy_pred)

# Visualize the confusion matrix

sns.heatmap(rcm, annot=True, cmap="Blues", fmt="g")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[ ]:





# In[51]:


# 8 Machine learning Algorithms will be applied to the dataset

classifiers = [[XGBClassifier(), 'XGB Classifier'],
               [RandomForestClassifier(), 'Random forest'],
               [KNeighborsClassifier(), 'K-Nearest Neighbors'],
               [SGDClassifier(), 'SGD Classifier'],
               [SVC(), 'SVC'],
               [GaussianNB(), "Naive Bayes"],
               [DecisionTreeClassifier(random_state = 42), "Decision tree"],
               [LogisticRegression(), 'Logistics Regression']
              ]


# In[38]:


acc_list = {}
precision_list = {}
recall_list = {}
roc_list = {}

for classifier in classifiers:
    model = classifier[0]
    model.fit(X_train, y_train)
    model_name = classifier[1]
    
    pred = model.predict(X_test)
    
    a_score = accuracy_score(y_test, pred)
    p_score = precision_score(y_test, pred)
    r_score = recall_score(y_test, pred)
    roc_score = roc_auc_score(y_test, pred)
    
    acc_list[model_name] = ([str(round(a_score*100, 2)) + '%'])
    precision_list[model_name] = ([str(round(p_score*100, 2)) + '%'])
    recall_list[model_name] = ([str(round(r_score*100, 2)) + '%'])
    roc_list[model_name] = ([str(round(roc_score*100, 2)) + '%'])
    
    if model_name != classifiers[-1][1]:
        print('')


# In[39]:


print("Accuracy Score")
s1 = pd.DataFrame(acc_list)
s1.head()


# In[40]:


print("Precision Score")
s2 = pd.DataFrame(precision_list)
s2.head()


# In[41]:


print("Recall Score")
s3 = pd.DataFrame(recall_list)
s3.head()


# In[42]:


print("ROC Score")
s4 = pd.DataFrame(roc_list)
s4.head()


# ### In Conclusion
# 
# From the analysis above, Logistic Regression performed better than Random Forest Classifier with accurancy of 86.89% and precision of 87.5%.

# In[ ]:




