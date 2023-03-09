#!/usr/bin/env python
# coding: utf-8

# ## **Importing libraries**

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from tabulate import tabulate
from six import StringIO
import missingno as msno


# ## **Loading dataset**

# In[4]:


dataset = pd.read_csv("aetna.csv")
dataset.head()


# **Dataset transformation**

# In[5]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[6]:


# Encoding Categorical Labels
dataset['Gender'] = encoder.fit_transform(dataset['Gender'])
dataset['Time_Since_Last_Dental_Checkup'] = encoder.fit_transform(dataset['Time_Since_Last_Dental_Checkup'])
dataset['Previous_Major_Dental_Procedure'] = encoder.fit_transform(dataset['Previous_Major_Dental_Procedure'])


# In[7]:


print("Based on co-relation, the attributes [ID, Previously_Insured] will be dropped")
print("-------------------------------------------------------------------------------")
final_dataset=dataset.drop(labels=["id", "Previously_Insured"],axis=1)
final_dataset.head()
# Tegan, why would we drop Previously_Insured here?


# **Dataset class balancing - SMOTE**

# In[8]:


X= final_dataset.drop(['Response'],axis=1)
y= final_dataset['Response']


# In[9]:


from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=33)
X_smote, y_smote = smote.fit_resample(X, y)


# In[10]:


print("Shape of Dataset after SMOTE : ", X_smote.shape,y_smote.shape)


# In[11]:


ch = pd.DataFrame(y_smote)
ch.head()


# In[12]:


ch_X = pd.DataFrame(X_smote)


# In[13]:


ch_dataset = pd.merge(ch_X,ch,right_index=True, left_index=True)
ch_dataset.head(5)


# In[14]:


print("Dataset shape after balancing: ", ch_dataset.shape)


# **Dataset splitting**

# In[15]:


X = ch_dataset.drop(["Response"],axis=1)
y = ch_dataset["Response"]


# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)


# In[17]:


print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)


# **Model Building - Random Forest**  

# In[18]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()


# In[19]:


# Train Random Forest Classifer
model.fit(X_train,y_train)


# **Model Evaluation - Random Forest**

# In[20]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[21]:


#Train Accuracy
X_train_predict = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_predict, y_train)
print("Train accuracy: ", training_data_accuracy)


# In[22]:


# Test Accuracy
y_pred = model.predict(X_test)
y=y_test
print("Test Accuracy:",accuracy_score(y, y_pred))


# In[23]:


from sklearn.metrics import confusion_matrix,classification_report
print("\n Classification report\n")
print(classification_report(y,y_pred))


# In[24]:


cm = confusion_matrix(y,y_pred)
print("Confusion matrix\n")
sns.heatmap(cm, annot=True)


# **Model Building - Logistic Regression**  

# In[25]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[26]:


# Train Logistic Regression
model.fit(X_train,y_train)


# **Model Evaluation - Logistic Regression**

# In[27]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[28]:


#Train Accuracy
X_train_predict = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_predict, y_train)
print("Train accuracy: ", training_data_accuracy)


# In[29]:


# Test Accuracy
y_pred = model.predict(X_test)
y=y_test
print("Test Accuracy:",accuracy_score(y, y_pred))


# In[30]:


from sklearn.metrics import confusion_matrix,classification_report
print("\n Classification report\n")
print(classification_report(y,y_pred))


# In[31]:


cm = confusion_matrix(y,y_pred)
print("Confusion matrix\n")
sns.heatmap(cm, annot=True)


# **Model Building - XGBoost**  

# In[32]:


import xgboost
from xgboost import XGBClassifier
model = XGBClassifier()


# In[33]:


# Train XGBoost
model.fit(X_train,y_train)


# **Model Evaluation - XGBoost**

# In[34]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[35]:


#Train Accuracy
X_train_predict = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_predict, y_train)
print("Train accuracy: ", training_data_accuracy)


# In[36]:


# Test Accuracy
y_pred = model.predict(X_test)
y=y_test
print("Test Accuracy:",accuracy_score(y, y_pred))


# In[37]:


from sklearn.metrics import confusion_matrix,classification_report
print("\n Classification report\n")
print(classification_report(y,y_pred))


# In[38]:


cm = confusion_matrix(y,y_pred)
print("Confusion matrix\n")
sns.heatmap(cm, annot=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




