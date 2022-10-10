#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/9characters/Regression-Performance-Analysis/blob/master/Lattice_Main.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# #Upload Data and Required python files

# #Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import os

from display import *
from helper_revised import *


# #Importing and Processing Data

# In[2]:


columns = ["compound", "ra", "rb", "fe", "bg", "a", "b", "c", "shape"]
inputs = {0:"ra", 1:"rb", 2:"fe", 3:"bg"}
mapping = {0:"a", 1:"b", 2:"c"}

training_data = pd.read_csv("training_data.csv")
testing_data = pd.read_csv("testing_data.csv")
# data.columns = columns


# In[3]:


testing_data.head()


# In[4]:


X_train, y_train = training_data.iloc[:, 1:5].dropna().values, training_data.iloc[:, 5:-1].dropna().values
X_test, y_test = testing_data.iloc[:, 1:5].dropna().values, testing_data.iloc[:, 5:-1].dropna().values


# In[5]:


#Dictionaries to store the r2_scores from different regressors
r2_scores = {}
predictions = {}


# In[6]:


# Assigining the regressor and working with it


# In[7]:


'''
NOTE:
Random Forest = RF
Decision Tree = DT
Linear Regression = LR
K Nearest Neighbour = KNN
Linear SVR = LSVR
Polynomial SVR = PSVR
RBF Kernel SVR = RBFSVR
'''


# In[8]:


#Creating a Plots folder to store the plots
if not os.path.exists("Plots"):
    os.mkdir("Plots")


# In[9]:


r_models = ["ANN", "RF", "DT", "KNN", "RBFSVR"]


# In[10]:


for r_model in r_models:
    #Prompt to get the regression model
    prompt = r_model

    y_pred, regressor, model, r2_score, coffs, intercepts = train_data(X_train, X_test, y_train, y_test, prompt)
    print(f"Optimal R2 Score for {regressor} is {r2_score}")

    #Storing the r2_score and predictions in the dictionaries
    r2_scores[regressor] = r2_score
    predictions[regressor] = y_pred
#     test_indices[regressor] = test_index

    # Analysis for Random Forest Regressor:
    if prompt == "RF":
        print("\nFeature Importance:")
        for i,c in enumerate(columns[1:-4]):
            print(f"{c}:\t{model.feature_importances_[i].round(3)}")

    # Analysis for Linear Regressor:
    if prompt == "LR":
        coffs = np.round(model.coef_, 3)
        intercepts = np.round(model.intercept_, 3)
        
        print("\nApproximated Equations from Linear Model:")
        for i in mapping:
            print(f"{mapping[i]} = {coffs[i][0]}{inputs[0]} + {coffs[i][1]}{inputs[1]} + {coffs[i][2]}{inputs[2]} + {coffs[i][3]}{inputs[3]} + {intercepts[i]}")

    # Analysis for Support Vector Regressor:
    if "SVR" in prompt:
        print(f"\nApproximated Equations from {regressor} Model: ")
        for i in mapping:
            print(f"{mapping[i]} = {coffs[i][0][0]}{inputs[0]} + {coffs[i][0][1]}{inputs[1]} + {coffs[i][0][2]}{inputs[2]} + {coffs[i][0][3]}{inputs[3]} + {intercepts[i]}")
    
    #Display Plots
    exp_vs_pred_subplots(y_test, y_pred, regressor)

    exp_vs_pred_lc(y_test, y_pred, regressor, r2_score)

    names = X_test[:, 0]
    exp_vs_pred_plotly(y_test, y_pred, regressor, names)


# #Comparision of performance of all regression models

# In[11]:


comparision(r2_scores)


# #Calculating PAD and Predicted value for Table

# In[12]:


prediction_table = pd.concat([
      pd.DataFrame(v, columns=['a', 'b', 'c'], index=np.repeat(k, len(v))) 
     for k, v in predictions.items()
  ]
).rename_axis('algorithm').reset_index()


# In[13]:


testing_compounds = list(testing_data['compound'])*5


# In[14]:


prediction_table['Compounds'] = testing_compounds


# In[15]:


prediction_table


# In[16]:


if not os.path.exists('Prediction_table'):
    os.mkdir("Prediction_table")


# In[17]:


prediction_table[ prediction_table["algorithm"] == "ANN" ].to_csv("Prediction_table/ANN.csv", index=False)


# In[18]:


prediction_table[ prediction_table["algorithm"] == "KNN" ].to_csv("Prediction_table/KNN.csv", index=False)


# In[19]:


prediction_table[ prediction_table["algorithm"] == "Random Forest" ].to_csv("Prediction_table/Random Forest.csv", index=False)


# In[20]:


prediction_table[ prediction_table["algorithm"] == "Decision Tree" ].to_csv("Prediction_table/Decision Tree.csv", index=False)


# In[21]:


prediction_table[ prediction_table["algorithm"] == "RBF Kernel SVR" ].to_csv("Prediction_table/RBF Kernel SVR.csv", index=False)


# In[22]:


if not os.path.exists("Pad_table"):
    os.mkdir("Pad_table")


# In[23]:


pad = lambda y_true, y_pred: ( np.abs(y_true - y_pred) / y_true) * 100


# In[24]:


pads = {}
for algo, y_pred in predictions.items():
    pads[algo] = pad(y_test, y_pred)
    
pad_table = pd.concat([
       pd.DataFrame(v, columns=['a', 'b', 'c'], index=np.repeat(k, len(v))) 
       for k, v in pads.items()
  ]
).rename_axis('algorithm').reset_index()


# In[25]:


pad_table['Compounds'] = testing_compounds


# In[26]:


pad_table.head(197)


# In[27]:


pad_table[ pad_table["algorithm"] == "ANN" ].loc[:, ["a", "b", "c"]].astype(np.float64).describe()


# In[28]:


pad_table[ pad_table["algorithm"] == "ANN" ].to_csv("Pad_table/ANN.csv", index=False)


# In[29]:


pad_table[ pad_table["algorithm"] == "KNN" ].loc[:, ["a", "b", "c"]].astype(np.float64).describe()


# In[30]:


pad_table[ pad_table["algorithm"] == "KNN" ].to_csv("Pad_table/KNN.csv", index=False)


# In[31]:


pad_table[ pad_table["algorithm"] == "Random Forest" ].loc[:, ["a", "b", "c"]].astype(np.float64).describe()


# In[32]:


pad_table[ pad_table["algorithm"] == "Random Forest" ].to_csv("Pad_table/Random Forest.csv",index=False)


# In[33]:


pad_table[ pad_table["algorithm"] == "Decision Tree" ].loc[:, ["a", "b", "c"]].astype(np.float64).describe()


# In[34]:


pad_table[ pad_table["algorithm"] == "Decision Tree" ].to_csv("Pad_table/Decision Tree.csv",index=False)


# In[35]:


pad_table[ pad_table["algorithm"] == "RBF Kernel SVR" ].loc[:, ["a", "b", "c"]].astype(np.float64).describe()


# In[36]:


pad_table[ pad_table["algorithm"] == "RBF Kernel SVR" ].to_csv("Pad_table/RBF Kernel SVR.csv",index=False)

