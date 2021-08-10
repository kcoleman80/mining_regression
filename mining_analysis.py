#!/usr/bin/env python
# coding: utf-8

# In[1]:


#league of legends 
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# read in the data
file = "/Users/kellymclean/2020_Dec_Kaggle_Mining/MiningProcess_Flotation_Plant_Database.csv"
df = pd.read_csv(file, low_memory=False)
df.head()


# In[3]:


df = pd.read_csv("/Users/kellymclean/2020_Dec_Kaggle_Mining/MiningProcess_Flotation_Plant_Database.csv",decimal=",",parse_dates=["date"],infer_datetime_format=True).drop_duplicates()


# In[4]:


df.shape


# In[5]:


df = df.dropna()
df.shape


# In[6]:


df.describe()


# In[7]:


plt.figure(figsize=(30, 25))
p = sns.heatmap(df.corr(), annot=True)


# In[8]:


df = df.drop(['date', '% Iron Concentrate', 'Ore Pulp pH', 'Flotation Column 01 Air Flow', 'Flotation Column 02 Air Flow', 'Flotation Column 03 Air Flow'], axis=1)


# In[9]:


df.head()


# In[10]:


Y = df['% Silica Concentrate']
X = df.drop(['% Silica Concentrate'], axis=1)


# In[11]:


#SCALING THE FEATURES
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

X_scaled = pd.DataFrame(min_max_scaler.fit_transform(X), columns=X.columns)


# In[12]:


#SPLITTING THE DATA
from sklearn.model_selection import train_test_split


# In[13]:


X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)


# # Training the Model 

# In[14]:


#linear regression
from sklearn.linear_model import LinearRegression


# In[15]:


reg = LinearRegression()


# In[16]:


_ = reg.fit(X_train, Y_train)


# In[17]:


predictions = reg.predict(X_test)
predictions


# In[18]:


#Finding Mean Squared Error
from sklearn.metrics import mean_squared_error


# In[19]:


error = mean_squared_error(Y_test, predictions)
error


# In[20]:


#Using Stochastic Gradient Descent
from sklearn.linear_model import SGDRegressor


# In[21]:


reg_sgd = SGDRegressor(max_iter=1000, tol=1e-3)


# In[22]:


_ = reg_sgd.fit(X_train, Y_train)


# In[23]:


predictions_sgd = reg_sgd.predict(X_test)


# In[24]:


#Finding Mean Squared Error of SGD
error_sgd = mean_squared_error(Y_test, predictions_sgd)
error_sgd

