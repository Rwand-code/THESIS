#!/usr/bin/env python
# coding: utf-8

# ## Preparing data and packages

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv(r'C:\Users\rwand\OneDrive\Desktop\THESIS\cell2cell.csv')


# ## Exploratory Data Analysis 

# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


## shows that there are multiple type of objects in the dataset
df.dtypes


# In[6]:


## Shows that the data is on different scales
df.describe()


# In[7]:


plt.figure(figsize=(10,5))
plt.subplot(1,1,1)
sn.distplot(df['MonthlyRevenue'])


# In[8]:


sn.boxplot(df['MonthlyRevenue'])


# In[9]:


## every NaN per column 
df.isnull().sum().sort_values(ascending = False)

## it seems like the majority of the columns have no missing data. This
## is misleading however, because there are categorical values in 
## the dataset marked 'unknown'. 

## ##https://stackoverflow.com/questions/26266362/how-to-count-the-nan-values-in-a-column-in-pandas-dataframe


# In[10]:


def print_unique_col_values(df):
       for column in df:
                print(f'{column}: {df[column].unique()}') 

print_unique_col_values(df)

# This shows the multiple columns with 'unknown' values for the
# categorical variables. 


# In[11]:


hsp = len(df[df.HandsetPrice == 'Unknown'])
ms = len(df[df.MaritalStatus == 'Unknown'])

print(hsp/len(df))
print(ms/len(df))

## This example shows the unknown values for the HandsetType feature.
## It shows that 28982 out of 51047 rows are unknown. This was not 
## clear before. More on handling this at the cleaning section.


# In[12]:


# ## This plot shows the imbalance in the target variables in the data.
#df.Churn.value_counts().plot(kind='bar', figsize=(12, 8))
sn.countplot(data = df, x='Churn', order=df.Churn.value_counts().index)
plt.xlabel("Churning")
plt.ylabel("Count")
plt.title("Class label balance");

# https://stackoverflow.com/questions/31460146/plotting-value-counts-in-seaborn-barplot


# In[13]:


## exact numbers and percentages of class imbalance
print(df.Churn.value_counts())
print(round(df.Churn.value_counts()/df.shape[0] * 100))


# ## Cleaning

# In[14]:


df_clean = df.copy()


# In[15]:


## First dropping CustomerID as its irrelevant to predicting. ServiceArea is encrypted in a way
## that we can not make inference from the values. 
df_clean.drop('CustomerID', axis = 'columns', inplace = True)
df_clean.drop('ServiceArea', axis = 'columns', inplace = True)
df_clean.drop('HandsetPrice', axis = 'columns', inplace = True)
df_clean.drop('MaritalStatus', axis = 'columns', inplace = True)


# In[16]:


## No duplicated rows in this dataset
df_clean.duplicated().sum()


# ### Handling missing values

# In[17]:


percent_missing = df_clean.isnull().sum() * 100 / len(df_clean)
missing_value_df_clean = pd.DataFrame({'column_name': df_clean.columns,
                                       'percent_missing': percent_missing})

##https://www.thiscodeworks.com/python-find-out-the-percentage-of-missing-values-in-each-column-in-the-given-dataset-stack-overflow-python/607d4c3f6013b5001411542c


# In[18]:


mis_rows = sum([True for idx,row in df.iterrows() if any(row.isnull())])
mis_rows/len(df)

#https://stackoverflow.com/questions/28199524/best-way-to-count-the-number-of-rows-with-missing-values-in-a-pandas-dataframe


# In[19]:


## Percentage missing per column is low
missing_value_df_clean.sort_values(by = ['percent_missing'], ascending = False)

# Because of the low percentage of missing data, these missing rows can be dropped.


# In[20]:


df_clean.shape


# In[21]:


## Dropping the rows with NaN values
df_clean.dropna(how = 'any', inplace = True)
df_clean.shape


# In[22]:


df.info()


# In[23]:


## Replacing feature values with numbers

df_clean.replace(to_replace = {"No" : 0, "Yes" : 1}, inplace = True)
df_clean.Homeownership.replace(to_replace = {"Unknown": 0, "Known": 1,}, inplace = True)


# ### Handling outliers

# In[24]:


## Everything above the 99th percentile will be replaced by the median value. This is only done
## for continuous features.

for i in df_clean.columns:
    print(round(df_clean.quantile(0.99)))
    print(round(df_clean.quantile(0.50)))


# In[25]:


df_clean['MonthlyRevenue'] = np.where(df_clean['MonthlyRevenue'] > 225, 48, df_clean['MonthlyRevenue'])
df_clean['MonthlyMinutes'] = np.where(df_clean['MonthlyMinutes'] > 2450, 366, df_clean['MonthlyMinutes'])
df_clean['TotalRecurringCharge'] = np.where(df_clean['TotalRecurringCharge'] > 120, 45, df_clean['TotalRecurringCharge'])
df_clean['DirectorAssistedCalls'] = np.where(df_clean['DirectorAssistedCalls'] > 10, 0, df_clean['DirectorAssistedCalls'])
df_clean['OverageMinutes'] = np.where(df_clean['OverageMinutes'] > 427, 3, df_clean['OverageMinutes'])
df_clean['RoamingCalls'] = np.where(df_clean['RoamingCalls'] > 21, 0, df_clean['RoamingCalls'])
df_clean['PercChangeMinutes'] = np.where(df_clean['PercChangeMinutes'] > 736, -5, df_clean['PercChangeMinutes'])
df_clean['PercChangeRevenues'] = np.where(df_clean['PercChangeRevenues'] > 114, 0, df_clean['PercChangeRevenues'])
df_clean['DroppedCalls'] = np.where(df_clean['DroppedCalls'] > 42, 3, df_clean['DroppedCalls'])
df_clean['BlockedCalls'] = np.where(df_clean['BlockedCalls'] > 47, 1, df_clean['BlockedCalls'])
df_clean['UnansweredCalls'] = np.where(df_clean['UnansweredCalls'] > 179, 16, df_clean['UnansweredCalls'])
df_clean['CustomerCareCalls'] = np.where(df_clean['CustomerCareCalls'] > 21, 0, df_clean['CustomerCareCalls'])
df_clean['ThreewayCalls'] = np.where(df_clean['ThreewayCalls'] > 4, 0, df_clean['ThreewayCalls'])
df_clean['ReceivedCalls'] = np.where(df_clean['ReceivedCalls'] > 775, 48, df_clean['ReceivedCalls'])
df_clean['OutboundCalls'] = np.where(df_clean['OutboundCalls'] > 162, 14, df_clean['OutboundCalls'])
df_clean['InboundCalls'] = np.where(df_clean['InboundCalls'] > 77, 2, df_clean['InboundCalls'])
df_clean['PeakCallsInOut'] = np.where(df_clean['PeakCallsInOut'] > 492, 48, df_clean['PeakCallsInOut'])
df_clean['OffPeakCallsInOut'] = np.where(df_clean['OffPeakCallsInOut'] > 436, 48, df_clean['OffPeakCallsInOut'])
df_clean['DroppedBlockedCalls'] = np.where(df_clean['DroppedBlockedCalls'] > 72, 5, df_clean['DroppedBlockedCalls'])
df_clean['CallForwardingCalls'] = np.where(df_clean['CallForwardingCalls'] > 0, 0, df_clean['CallForwardingCalls'])
df_clean['CallWaitingCalls'] = np.where(df_clean['CallWaitingCalls'] > 23, 0, df_clean['CallWaitingCalls'])
df_clean['MonthsInService'] = np.where(df_clean['MonthsInService'] > 49, 16, df_clean['MonthsInService'])
df_clean['UniqueSubs'] = np.where(df_clean['UniqueSubs'] > 5, 1, df_clean['UniqueSubs'])
df_clean['ActiveSubs'] = np.where(df_clean['ActiveSubs'] > 4, 1, df_clean['ActiveSubs'])
df_clean['CurrentEquipmentDays'] = np.where(df_clean['CurrentEquipmentDays'] > 1138, 329, df_clean['CurrentEquipmentDays'])


# In[ ]:





# ## Dummy coding categorical variables

# In[26]:


df_dum = pd.get_dummies(df_clean)


# In[27]:


df_dum.info()


# In[ ]:





# ## Splitting

# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


## Splitting the data in training, test and validation. 

X = df_dum.drop('Churn', axis = 'columns')
y = df_dum.Churn

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42) 


# ## Normalizing

# In[30]:


from sklearn.preprocessing import MinMaxScaler


# In[31]:


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.fit_transform(X_val)
X_test_scaled = scaler.transform(X_test)


# In[32]:


print(X_train_scaled.shape)
print(X_val_scaled.shape)
print(X_test_scaled.shape)


# ## Feature selection mutual infromation

# In[33]:


## Feature selection
from sklearn.feature_selection import SelectKBest, mutual_info_classif
np.random.seed(42)
selector = SelectKBest(mutual_info_classif, k=20)

X_train_mi = selector.fit_transform(X_train_scaled, y_train)
X_val_mi = selector.transform(X_val_scaled)
X_test_mi = selector.transform(X_test_scaled)
y_train.shape, X_train_mi.shape, X_val_mi.shape, X_test_mi.shape


# In[34]:


X.columns[selector.get_support()]


# In[35]:


print('Original number of train features:', X.shape)
print('Reduced number of features:', X_train.shape)


# # Resampling techniques

# In[36]:


## importing resampling packages
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from collections import Counter


# ## Resampling methods

# ### Undersampling - no feature selection

# In[37]:


rus = RandomUnderSampler(random_state=42, replacement=True)# fit predictor and target variable
X_rus, y_rus = rus.fit_resample(X_train_scaled, y_train)

print('original dataset shape:', Counter(y))
print('Resample dataset shape', Counter(y_rus))

#https://www.analyticsvidhya.com/blog/2020/07/10-techniques-to-deal-with-class-imbalance-in-machine-learning/


# ### Undersampling - mutual information feature selection

# In[38]:


rus = RandomUnderSampler(random_state=42, replacement=True)# fit predictor and target variable
X_rus_mi, y_rus_mi = rus.fit_resample(X_train_mi, y_train)

print('original dataset shape:', Counter(y))
print('Resample dataset shape', Counter(y_rus))

#https://www.analyticsvidhya.com/blog/2020/07/10-techniques-to-deal-with-class-imbalance-in-machine-learning/


# ### Random oversampling - no feature selection

# In[39]:


ros = RandomOverSampler(random_state=42)
X_ros, y_ros = ros.fit_resample(X_train_scaled, y_train)

# fit predictor and target variablex_ros, y_ros = ros.fit_resample(x, y)

print('Original dataset shape', Counter(y))
print('Resample dataset shape', Counter(y_ros))

#https://www.analyticsvidhya.com/blog/2020/07/10-techniques-to-deal-with-class-imbalance-in-machine-learning/


# ### Random oversampling - mutual information feature selection

# In[40]:


ros = RandomOverSampler(random_state=42)
X_ros_mi, y_ros_mi = ros.fit_resample(X_train_mi, y_train)

# fit predictor and target variablex_ros, y_ros = ros.fit_resample(x, y)

print('Original dataset shape', Counter(y))
print('Resample dataset shape', Counter(y_ros))

#https://www.analyticsvidhya.com/blog/2020/07/10-techniques-to-deal-with-class-imbalance-in-machine-learning/


# ### Synthetic Minority Oversampling Technique (SMOTE) - no feature selection

# In[41]:


smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X_train_scaled, y_train)

print('Original dataset shape', Counter(y))
print('Resample dataset shape', Counter(y_smote))

#https://www.analyticsvidhya.com/blog/2020/07/10-techniques-to-deal-with-class-imbalance-in-machine-learning/


# ### Synthetic Minority Oversampling Technique (SMOTE) - mutual information feature selection

# In[42]:


smote = SMOTE()
X_smote_mi, y_smote_mi = smote.fit_resample(X_train_mi, y_train)

print('Original dataset shape', Counter(y))
print('Resample dataset shape', Counter(y_smote))

#https://www.analyticsvidhya.com/blog/2020/07/10-techniques-to-deal-with-class-imbalance-in-machine-learning/


# ## Feedforward Neural networks - no feature selection

# In[43]:


pip install tensorflow_addons


# In[44]:


import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
from keras import regularizers
from keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
import tensorflow_addons as tfa

from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from numpy.random import seed


# In[45]:


seed(42)
tf.random.set_seed(42)


# ### Untuned and simple architecture as baseline

# In[46]:


fnn = Sequential()
fnn.add(Dense(64, activation = 'relu', input_dim=69)) ## 69 representing all features
fnn.add(Dense(32, activation = 'relu'))
fnn.add(Dense(1, activation = 'sigmoid'))
fnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy', keras.metrics.AUC()])


# In[47]:


fnn.summary()


# ### Fitting with no sampling methods  

# In[48]:


baseline = fnn.fit(X_train_scaled, y_train, batch_size=32, epochs=50, 
               validation_data=(X_val_scaled, y_val))


# In[49]:


plt.style.use('seaborn')
plt.figure(figsize = (16,5))
plt.subplot(121)
plt.title('loss with epochs', fontsize=16)
plt.plot(baseline.history['loss'], marker='o', color="orange", label="Training loss")
plt.plot(baseline.history['val_loss'], marker='o',color="blue", label="Validation loss")
plt.xticks()
plt.legend()

plt.subplot(122)
plt.title('accuracy with epochs', fontsize=16)
plt.plot(baseline.history['accuracy'], marker='o', color="orange", label="Training Acc")
plt.plot(baseline.history['val_accuracy'], marker='o',color="blue", label="Validation Acc")
plt.xticks()
plt.legend()


# In[50]:


pred_baseval = fnn.predict(X_val_scaled, verbose = 1).ravel()
pred_basetest = fnn.predict(X_test_scaled, verbose = 1).ravel()


# In[51]:


fpr_val, tpr_val, thresholds_val = roc_curve(y_val, pred_baseval)
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, pred_basetest)


# In[52]:


auc_val = auc(fpr_val, tpr_val)
auc_test = auc(fpr_test, tpr_test)


# In[53]:


plt.plot(fpr_val, tpr_val, color='orange',
         lw=1.5, label='AUC = %0.2f' % auc_val)
plt.plot([0, 1], [0, 1], color='blue', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Baseline Validation')
plt.legend(loc="lower right");
plt.show()

plt.plot(fpr_test, tpr_test, color='orange',
         lw=1.5, label='AUC = %0.2f' % auc_test)
plt.plot([0, 1], [0, 1], color='blue', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Baseline Test')
plt.legend(loc="lower right");

#confusion matrix for validation
conf_val = confusion_matrix(y_val, pred_baseval.round())
conf_test = confusion_matrix(y_test, pred_basetest.round())

plt.figure(figsize=(12,6))
plt.subplot(121)
sn.heatmap(conf_val, annot = True, fmt=".3f", square=True, cmap = "Blues_r");
plt.ylabel('True label');
plt.xlabel('Predicted label')
plt.title('Confusion matrix - Baseline Validation')
plt.subplot(122)
sn.heatmap(conf_test, annot = True, fmt=".3f", square = True, cmap = "Blues_r");
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix - Baseline Test')
plt.savefig('Confusion matrix - Baseline Validation and Test')
plt.show()


# In[54]:


metrics = fnn.evaluate(X_test_scaled, y_test)
metrics


# ## Fitting with Random Undersampling

# In[55]:


X_train_rus_scaled = scaler.fit_transform(X_rus)

rus_base = fnn.fit(X_train_rus_scaled, y_rus, batch_size=32, epochs=50, 
validation_data=(X_val_scaled, y_val))


# In[56]:


plt.style.use('seaborn')
plt.figure(figsize = (16,5))
plt.subplot(121)
plt.title('loss with epochs', fontsize=16)
plt.plot(rus_base.history['loss'], marker='o', color="orange", label="Training loss")
plt.plot(rus_base.history['val_loss'], marker='o',color="blue", label="Validation loss")
plt.xticks()
plt.legend()

plt.subplot(122)
plt.title('accuracy with epochs', fontsize=16)
plt.plot(rus_base.history['accuracy'], marker='o', color="orange", label="Training Acc")
plt.plot(rus_base.history['val_accuracy'], marker='o',color="blue", label="Validation Acc")
plt.xticks()
plt.legend()


# In[57]:


pred_rus_baseval = fnn.predict(X_val_scaled, verbose = 1).ravel()
pred_rus_basetest = fnn.predict(X_test_scaled, verbose = 1).ravel()

fpr_val, tpr_val, thresholds_val = roc_curve(y_val, pred_rus_baseval)
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, pred_rus_basetest)

auc_val = auc(fpr_val, tpr_val)
auc_test = auc(fpr_test, tpr_test)

plt.plot(fpr_val, tpr_val, color='orange',
         lw=1.5, label='AUC = %0.2f' % auc_val)
plt.plot([0, 1], [0, 1], color='blue', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Baseline Validation')
plt.legend(loc="lower right");
plt.show()

plt.plot(fpr_test, tpr_test, color='orange',
         lw=1.5, label='AUC = %0.2f' % auc_test)
plt.plot([0, 1], [0, 1], color='blue', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Baseline Test')
plt.legend(loc="lower right");

#confusion matrix for validation
conf_val = confusion_matrix(y_val, pred_rus_baseval.round())
conf_test = confusion_matrix(y_test, pred_rus_basetest.round())

plt.figure(figsize=(12,6))
plt.subplot(121)
sn.heatmap(conf_val, annot = True, fmt=".3f", square=True, cmap = "Blues_r");
plt.ylabel('True label');
plt.xlabel('Predicted label')
plt.title('Confusion matrix - Baseline Validation')
plt.subplot(122)
sn.heatmap(conf_test, annot = True, fmt=".3f", square = True, cmap = "Blues_r");
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix - Baseline Test')
plt.savefig('Confusion matrix - Baseline Validation and Test')
plt.show()

metrics = fnn.evaluate(X_test_scaled, y_test)
metrics


# ## Fitting with Random Oversampling

# In[58]:


X_train_ros_scaled = scaler.fit_transform(X_ros)

ros_base = fnn.fit(X_train_ros_scaled, y_ros, batch_size=32, epochs=50, 
               validation_data=(X_val_scaled, y_val)) 


# In[59]:


plt.style.use('seaborn')
plt.figure(figsize = (16,5))
plt.subplot(121)
plt.title('loss with epochs', fontsize=16)
plt.plot(ros_base.history['loss'], marker='o', color="orange", label="Training loss")
plt.plot(ros_base.history['val_loss'], marker='o',color="blue", label="Validation loss")
plt.xticks()
plt.legend()

plt.subplot(122)
plt.title('accuracy with epochs', fontsize=16)
plt.plot(ros_base.history['accuracy'], marker='o', color="orange", label="Training Acc")
plt.plot(ros_base.history['val_accuracy'], marker='o',color="blue", label="Validation Acc")
plt.xticks()
plt.legend()


# In[129]:


pred_ros_baseval = fnn.predict(X_val_scaled, verbose = 1).ravel()
pred_ros_basetest = fnn.predict(X_test_scaled, verbose = 1).ravel()

fpr_val, tpr_val, thresholds_val = roc_curve(y_val, pred_ros_baseval)
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, pred_ros_basetest)

auc_val = auc(fpr_val, tpr_val)
auc_test = auc(fpr_test, tpr_test)

plt.plot(fpr_val, tpr_val, color='orange',
         lw=1.5, label='AUC = %0.2f' % auc_val)
plt.plot([0, 1], [0, 1], color='blue', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Baseline Validation')
plt.legend(loc="lower right");
plt.show()

plt.plot(fpr_test, tpr_test, color='orange',
         lw=1.5, label='AUC = %0.2f' % auc_test)
plt.plot([0, 1], [0, 1], color='blue', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Baseline Test')
plt.legend(loc="lower right");

#confusion matrix for validation
conf_val = confusion_matrix(y_val, pred_ros_baseval.round())
conf_test = confusion_matrix(y_test, pred_ros_basetest.round())

plt.figure(figsize=(12,6))
plt.subplot(121)
sn.heatmap(conf_val, annot = True, fmt=".3f", square=True, cmap = "Blues_r");
plt.ylabel('True label');
plt.xlabel('Predicted label')
plt.title('Confusion matrix - Baseline Validation')
plt.subplot(122)
sn.heatmap(conf_test, annot = True, fmt=".3f", square = True, cmap = "Blues_r");
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix - Untuned oversampled Test')
plt.savefig('Confusion matrix - Baseline Validation and Test')
plt.show()

metrics = fnn.evaluate(X_test_scaled, y_test)
metrics


# ## Fitting with SMOTE

# In[61]:


X_train_smote_scaled = scaler.fit_transform(X_smote)

smote_base = fnn.fit(X_train_smote_scaled, y_smote, batch_size=32, epochs=50, 
               validation_data=(X_val_scaled, y_val))


# In[62]:


plt.style.use('seaborn')
plt.figure(figsize = (16,5))
plt.subplot(121)
plt.title('loss with epochs', fontsize=16)
plt.plot(smote_base.history['loss'], marker='o', color="orange", label="Training loss")
plt.plot(smote_base.history['val_loss'], marker='o',color="blue", label="Validation loss")
plt.xticks()
plt.legend()

plt.subplot(122)
plt.title('accuracy with epochs', fontsize=16)
plt.plot(smote_base.history['accuracy'], marker='o', color="orange", label="Training Acc")
plt.plot(smote_base.history['val_accuracy'], marker='o',color="blue", label="Validation Acc")
plt.xticks()
plt.legend()


# In[63]:


pred_smote_baseval = fnn.predict(X_val_scaled, verbose = 1).ravel()
pred_smote_basetest = fnn.predict(X_test_scaled, verbose = 1).ravel()

fpr_val, tpr_val, thresholds_val = roc_curve(y_val, pred_smote_baseval)
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, pred_smote_basetest)

auc_val = auc(fpr_val, tpr_val)
auc_test = auc(fpr_test, tpr_test)

plt.plot(fpr_val, tpr_val, color='orange',
         lw=1.5, label='AUC = %0.2f' % auc_val)
plt.plot([0, 1], [0, 1], color='blue', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Baseline Validation')
plt.legend(loc="lower right");
plt.show()

plt.plot(fpr_test, tpr_test, color='orange',
         lw=1.5, label='AUC = %0.2f' % auc_test)
plt.plot([0, 1], [0, 1], color='blue', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Baseline Test')
plt.legend(loc="lower right");

#confusion matrix for validation
conf_val = confusion_matrix(y_val, pred_smote_baseval.round())
conf_test = confusion_matrix(y_test, pred_smote_basetest.round())

plt.figure(figsize=(12,6))
plt.subplot(121)
sn.heatmap(conf_val, annot = True, fmt=".3f", square=True, cmap = "Blues_r");
plt.ylabel('True label');
plt.xlabel('Predicted label')
plt.title('Confusion matrix - Baseline Validation')
plt.subplot(122)
sn.heatmap(conf_test, annot = True, fmt=".3f", square = True, cmap = "Blues_r");
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix - Baseline Test')
plt.savefig('Confusion matrix - Baseline Validation and Test')
plt.show()

metrics = fnn.evaluate(X_test_scaled, y_test)
metrics


# ## Deep Feedforward Neural Networks with mutual information feature selection

# In[64]:


fnn_mi = Sequential()
fnn_mi.add(Dense(32, activation = 'relu', input_dim=20)) ## 20 representing subset of features
fnn_mi.add(Dense(16, activation = 'relu'))
fnn_mi.add(Dense(1, activation = 'sigmoid'))
fnn_mi.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=[keras.metrics.AUC(), 'accuracy'])


# In[65]:


fnn_mi.summary()


# ### Fitting with no sampling methods  

# In[66]:


baseline_mi = fnn_mi.fit(X_train_mi, y_train, batch_size=32, epochs=50, 
               validation_data=(X_val_mi, y_val))


# In[67]:


plt.style.use('seaborn')
plt.figure(figsize = (16,5))
plt.subplot(121)
plt.title('loss with epochs', fontsize=16)
plt.plot(baseline_mi.history['loss'], marker='o', color="orange", label="Training loss")
plt.plot(baseline_mi.history['val_loss'], marker='o',color="blue", label="Validation loss")
plt.xticks()
plt.legend()

plt.subplot(122)
plt.title('accuracy with epochs', fontsize=16)
plt.plot(baseline_mi.history['accuracy'], marker='o', color="orange", label="Training Acc")
plt.plot(baseline_mi.history['val_accuracy'], marker='o',color="blue", label="Validation Acc")
plt.xticks()
plt.legend()


# In[68]:


pred_baseval_mi = fnn_mi.predict(X_val_mi, verbose = 1).ravel()
pred_basetest_mi = fnn_mi.predict(X_test_mi, verbose = 1).ravel()


# In[69]:


fpr_val, tpr_val, thresholds_val = roc_curve(y_val, pred_baseval_mi)
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, pred_basetest_mi)


# In[70]:


auc_val = auc(fpr_val, tpr_val)
auc_test = auc(fpr_test, tpr_test)


# In[71]:


plt.plot(fpr_val, tpr_val, color='orange',
         lw=1.5, label='AUC = %0.2f' % auc_val)
plt.plot([0, 1], [0, 1], color='blue', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Baseline Validation')
plt.legend(loc="lower right");
plt.show()

plt.plot(fpr_test, tpr_test, color='orange',
         lw=1.5, label='AUC = %0.2f' % auc_test)
plt.plot([0, 1], [0, 1], color='blue', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Baseline Test')
plt.legend(loc="lower right");

#confusion matrix for validation
conf_val = confusion_matrix(y_val, pred_baseval.round())
conf_test = confusion_matrix(y_test, pred_basetest.round())

plt.figure(figsize=(12,6))
plt.subplot(121)
sn.heatmap(conf_val, annot = True, fmt=".3f", square=True, cmap = "Blues_r");
plt.ylabel('True label');
plt.xlabel('Predicted label')
plt.title('Confusion matrix - Baseline Validation')
plt.subplot(122)
sn.heatmap(conf_test, annot = True, fmt=".3f", square = True, cmap = "Blues_r");
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix - Baseline Test')
plt.savefig('Confusion matrix - Baseline Validation and Test')
plt.show()


# In[72]:


metrics = fnn_mi.evaluate(X_test_mi, y_test)
metrics


# In[ ]:





# ### Fitting with Random Undersampling

# In[73]:


X_train_rus_mi = scaler.fit_transform(X_rus_mi)

rus_base_mi = fnn_mi.fit(X_train_rus_mi, y_rus_mi, batch_size=32, epochs=50, 
validation_data=(X_val_mi, y_val))


# In[74]:


plt.style.use('seaborn')
plt.figure(figsize = (16,5))
plt.subplot(121)
plt.title('loss with epochs', fontsize=16)
plt.plot(rus_base_mi.history['loss'], marker='o', color="orange", label="Training loss")
plt.plot(rus_base_mi.history['val_loss'], marker='o',color="blue", label="Validation loss")
plt.xticks()
plt.legend()

plt.subplot(122)
plt.title('accuracy with epochs', fontsize=16)
plt.plot(rus_base_mi.history['accuracy'], marker='o', color="orange", label="Training Acc")
plt.plot(rus_base_mi.history['val_accuracy'], marker='o',color="blue", label="Validation Acc")
plt.xticks()
plt.legend()


# In[75]:


pred_rus_baseval_mi = fnn_mi.predict(X_val_mi, verbose = 1).ravel()
pred_rus_basetest_mi = fnn_mi.predict(X_test_mi, verbose = 1).ravel()

fpr_val, tpr_val, thresholds_val = roc_curve(y_val, pred_rus_baseval_mi)
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, pred_rus_basetest_mi)

auc_val = auc(fpr_val, tpr_val)
auc_test = auc(fpr_test, tpr_test)

plt.plot(fpr_val, tpr_val, color='orange',
         lw=1.5, label='AUC = %0.2f' % auc_val)
plt.plot([0, 1], [0, 1], color='blue', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Baseline Validation')
plt.legend(loc="lower right");
plt.show()

plt.plot(fpr_test, tpr_test, color='orange',
         lw=1.5, label='AUC = %0.2f' % auc_test)
plt.plot([0, 1], [0, 1], color='blue', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Baseline Test')
plt.legend(loc="lower right");

#confusion matrix for validation
conf_val = confusion_matrix(y_val, pred_rus_baseval_mi.round())
conf_test = confusion_matrix(y_test, pred_rus_basetest_mi.round())

plt.figure(figsize=(12,6))
plt.subplot(121)
sn.heatmap(conf_val, annot = True, fmt=".3f", square=True, cmap = "Blues_r");
plt.ylabel('True label');
plt.xlabel('Predicted label')
plt.title('Confusion matrix - Baseline Validation')
plt.subplot(122)
sn.heatmap(conf_test, annot = True, fmt=".3f", square = True, cmap = "Blues_r");
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix - Baseline Test')
plt.savefig('Confusion matrix - Baseline Validation and Test')
plt.show()

metrics = fnn_mi.evaluate(X_test_mi, y_test)
metrics


# ## Fitting with Random Oversampling

# In[76]:


X_train_ros_mi = scaler.fit_transform(X_ros_mi)

ros_base_mi = fnn_mi.fit(X_train_ros_mi, y_ros_mi, batch_size=32, epochs=50, 
               validation_data=(X_val_mi, y_val)) 


# In[77]:


plt.style.use('seaborn')
plt.figure(figsize = (16,5))
plt.subplot(121)
plt.title('loss with epochs', fontsize=16)
plt.plot(ros_base_mi.history['loss'], marker='o', color="orange", label="Training loss")
plt.plot(ros_base_mi.history['val_loss'], marker='o',color="blue", label="Validation loss")
plt.xticks()
plt.legend()

plt.subplot(122)
plt.title('accuracy with epochs', fontsize=16)
plt.plot(ros_base_mi.history['accuracy'], marker='o', color="orange", label="Training Acc")
plt.plot(ros_base_mi.history['val_accuracy'], marker='o',color="blue", label="Validation Acc")
plt.xticks()
plt.legend()


# In[78]:


pred_ros_baseval_mi = fnn_mi.predict(X_val_mi, verbose = 1).ravel()
pred_ros_basetest_mi = fnn_mi.predict(X_test_mi, verbose = 1).ravel()

fpr_val, tpr_val, thresholds_val = roc_curve(y_val, pred_ros_baseval_mi)
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, pred_ros_basetest_mi)

auc_val = auc(fpr_val, tpr_val)
auc_test = auc(fpr_test, tpr_test)

plt.plot(fpr_val, tpr_val, color='orange',
         lw=1.5, label='AUC = %0.2f' % auc_val)
plt.plot([0, 1], [0, 1], color='blue', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Baseline Validation')
plt.legend(loc="lower right");
plt.show()

plt.plot(fpr_test, tpr_test, color='orange',
         lw=1.5, label='AUC = %0.2f' % auc_test)
plt.plot([0, 1], [0, 1], color='blue', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Baseline Test')
plt.legend(loc="lower right");

#confusion matrix for validation
conf_val = confusion_matrix(y_val, pred_ros_baseval_mi.round())
conf_test = confusion_matrix(y_test, pred_ros_basetest_mi.round())

plt.figure(figsize=(12,6))
plt.subplot(121)
sn.heatmap(conf_val, annot = True, fmt=".3f", square=True, cmap = "Blues_r");
plt.ylabel('True label');
plt.xlabel('Predicted label')
plt.title('Confusion matrix - Baseline Validation')
plt.subplot(122)
sn.heatmap(conf_test, annot = True, fmt=".3f", square = True, cmap = "Blues_r");
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix - Baseline Test')
plt.savefig('Confusion matrix - Baseline Validation and Test')
plt.show()

metrics = fnn_mi.evaluate(X_test_mi, y_test)
metrics


# ## Fitting with SMOTE

# In[79]:


X_train_smote_mi = scaler.fit_transform(X_smote_mi)

smote_base_mi = fnn_mi.fit(X_train_smote_mi, y_smote_mi, batch_size=32, epochs=50,
                           validation_data=(X_val_mi, y_val))


# In[80]:


plt.style.use('seaborn')
plt.figure(figsize = (16,5))
plt.subplot(121)
plt.title('loss with epochs', fontsize=16)
plt.plot(smote_base_mi.history['loss'], marker='o', color="orange", label="Training loss")
plt.plot(smote_base_mi.history['val_loss'], marker='o',color="blue", label="Validation loss")
plt.xticks()
plt.legend()

plt.subplot(122)
plt.title('accuracy with epochs', fontsize=16)
plt.plot(smote_base_mi.history['accuracy'], marker='o', color="orange", label="Training Acc")
plt.plot(smote_base_mi.history['val_accuracy'], marker='o',color="blue", label="Validation Acc")
plt.xticks()
plt.legend()


# In[81]:


pred_smote_baseval_mi = fnn_mi.predict(X_val_mi, verbose = 1).ravel()
pred_smote_basetest_mi = fnn_mi.predict(X_test_mi, verbose = 1).ravel()

fpr_val, tpr_val, thresholds_val = roc_curve(y_val, pred_smote_baseval_mi)
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, pred_smote_basetest_mi)

auc_val = auc(fpr_val, tpr_val)
auc_test = auc(fpr_test, tpr_test)

plt.plot(fpr_val, tpr_val, color='orange',
         lw=1.5, label='AUC = %0.2f' % auc_val)
plt.plot([0, 1], [0, 1], color='blue', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Baseline Validation')
plt.legend(loc="lower right");
plt.show()

plt.plot(fpr_test, tpr_test, color='orange',
         lw=1.5, label='AUC = %0.2f' % auc_test)
plt.plot([0, 1], [0, 1], color='blue', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Baseline Test')
plt.legend(loc="lower right");

#confusion matrix for validation
conf_val = confusion_matrix(y_val, pred_smote_baseval_mi.round())
conf_test = confusion_matrix(y_test, pred_smote_basetest_mi.round())

plt.figure(figsize=(12,6))
plt.subplot(121)
sn.heatmap(conf_val, annot = True, fmt=".3f", square=True, cmap = "Blues_r");
plt.ylabel('True label');
plt.xlabel('Predicted label')
plt.title('Confusion matrix - Baseline Validation')
plt.subplot(122)
sn.heatmap(conf_test, annot = True, fmt=".3f", square = True, cmap = "Blues_r");
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix - Baseline Test')
plt.savefig('Confusion matrix - Baseline Validation and Test')
plt.show()

metrics = fnn_mi.evaluate(X_test_mi, y_test)
metrics


# In[ ]:





# In[ ]:





# # Build tuner for hyperparameter tuning

# In[82]:


def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', 2, 10)):
        model.add(keras.layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=32,
                                            max_value=128,
                                            step=32),
                               activation=hp.Choice('acticvation', ['relu', 'tanh', 'LeakyReLU'])))
    drop_rate = hp.Choice('drop_rate', [ 0.0, 0.1, 0.2, 0.3, 0.4,0.5 ])
    kernels = hp.Choice('l2', values=[0.01,0.001,0.1,0.005,0.05])
    model.add(keras.layers.Dropout(rate=drop_rate))
    kernel_regularizer=tf.keras.regularizers.l1(kernels)
    model.add(layers.Dense(1, activation='sigmoid'))
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[keras.metrics.AUC(), 'accuracy'])
    return model


# ### Tuning baseline 

# In[83]:


seed(42)
tf.random.set_seed(42)
## tuner for unsampled training set
tuner_baseline = kt.BayesianOptimization(
    build_model,
    objective=kt.Objective("auc", direction="max"),
    directory='directory',
    project_name='Tuned_parameters',
    max_trials=5,
    overwrite=True)


# In[84]:


seed(42)
tf.random.set_seed(42)

tuner_baseline.search(X_train_scaled, y_train, epochs=20, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',  
              patience=1)], validation_data=(X_val_scaled, y_val))
best_model1 = tuner_baseline.get_best_models()[0]


# In[85]:


tuner_baseline.results_summary()


# In[86]:


seed(42)
tf.random.set_seed(42)

base_models = tuner_baseline.get_best_models(num_models=3)
best_model_base = base_models[0]

best_model_base.build(input_shape=(39820, 69))
best_model_base.summary()

best_hps_base = tuner_baseline.get_best_hyperparameters(5)
# Build the model with the best hp.
model_base = build_model(best_hps_base[0])

baseline_tuned = model_base.fit(X_train_scaled, y_train, batch_size= 32, epochs = 50, 
          validation_data = (X_val_scaled, y_val))


# In[87]:


plt.style.use('seaborn')
plt.figure(figsize = (16,5))
plt.subplot(121)
plt.title('loss with epochs', fontsize=16)
plt.plot(baseline_tuned.history['loss'], marker='o', color="orange", label="Training loss")
plt.plot(baseline_tuned.history['val_loss'], marker='o',color="blue", label="Validation loss")
plt.xticks()
plt.legend()

plt.subplot(122)
plt.title('accuracy with epochs', fontsize=16)
plt.plot(baseline_tuned.history['accuracy'], marker='o', color="orange", label="Training Acc")
plt.plot(baseline_tuned.history['val_accuracy'], marker='o',color="blue", label="Validation Acc")
plt.xticks()
plt.legend()


# In[88]:


pred_tuneval = model_base.predict(X_val_scaled, verbose = 1).ravel()
pred_tunetest = model_base.predict(X_test_scaled, verbose = 1).ravel()

fpr_val, tpr_val, thresholds_val = roc_curve(y_val, pred_tuneval)
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, pred_tunetest)

auc_val = auc(fpr_val, tpr_val)
auc_test = auc(fpr_test, tpr_test)

plt.plot(fpr_val, tpr_val, color='orange',
         lw=1.5, label='AUC = %0.2f' % auc_val)
plt.plot([0, 1], [0, 1], color='blue', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Baseline Validation')
plt.legend(loc="lower right");
plt.show()

plt.plot(fpr_test, tpr_test, color='orange',
         lw=1.5, label='AUC = %0.2f' % auc_test)
plt.plot([0, 1], [0, 1], color='blue', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Baseline Test')
plt.legend(loc="lower right");

#confusion matrix for validation
conf_val = confusion_matrix(y_val, pred_tuneval.round())
conf_test = confusion_matrix(y_test, pred_tunetest.round())

plt.figure(figsize=(12,6))
plt.subplot(121)
sn.heatmap(conf_val, annot = True, fmt=".3f", square=True, cmap = "Blues_r");
plt.ylabel('True label');
plt.xlabel('Predicted label')
plt.title('Confusion matrix - Baseline Validation')
plt.subplot(122)
sn.heatmap(conf_test, annot = True, fmt=".3f", square = True, cmap = "Blues_r");
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix - Baseline Test')
plt.savefig('Confusion matrix - Baseline Validation and Test')
plt.show()

metrics = model_base.evaluate(X_test_scaled, y_test)
metrics


# ### Tuning random undersampler

# In[89]:


seed(42)
tf.random.set_seed(42)
## tuner for unsampled training set
tuner_rus = kt.BayesianOptimization(
    build_model,
    objective=kt.Objective("auc", direction="max"),
    directory='directory',
    project_name='Tuned_parameters',
    max_trials=5,
    overwrite=True)


# In[90]:


seed(42)
tf.random.set_seed(42)

tuner_rus.search(X_train_rus_scaled, y_rus, epochs=20, validation_data=(X_val_scaled, y_val))
best_model2 = tuner_rus.get_best_models()[0]


# In[91]:


# tuner.results_summary()


# In[92]:


seed(42)
tf.random.set_seed(42)

rus_models = tuner_rus.get_best_models(num_models=3)
best_model_rus = rus_models[0]

best_model_rus.build(input_shape=(39820, 69))
best_model_rus.summary()

best_hps_rus = tuner_rus.get_best_hyperparameters(5)
# Build the model with the best hp.
model_rus = build_model(best_hps_rus[0])

rus_tuned = model_rus.fit(X_train_rus_scaled, y_rus, batch_size= 32, epochs = 50, 
          validation_data = (X_val_scaled, y_val))


# In[93]:


plt.style.use('seaborn')
plt.figure(figsize = (16,5))
plt.subplot(121)
plt.title('loss with epochs', fontsize=16)
plt.plot(rus_tuned.history['loss'], marker='o', color="orange", label="Training loss")
plt.plot(rus_tuned.history['val_loss'], marker='o',color="blue", label="Validation loss")
plt.xticks()
plt.legend()

plt.subplot(122)
plt.title('accuracy with epochs', fontsize=16)
plt.plot(rus_tuned.history['accuracy'], marker='o', color="orange", label="Training Acc")
plt.plot(rus_tuned.history['val_accuracy'], marker='o',color="blue", label="Validation Acc")
plt.xticks()
plt.legend()


# In[94]:


pred_tuneval = model_rus.predict(X_val_scaled, verbose = 1).ravel()
pred_tunetest = model_rus.predict(X_test_scaled, verbose = 1).ravel()

fpr_val, tpr_val, thresholds_val = roc_curve(y_val, pred_tuneval)
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, pred_tunetest)

auc_val = auc(fpr_val, tpr_val)
auc_test = auc(fpr_test, tpr_test)

plt.plot(fpr_val, tpr_val, color='orange',
         lw=1.5, label='AUC = %0.2f' % auc_val)
plt.plot([0, 1], [0, 1], color='blue', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Baseline Validation')
plt.legend(loc="lower right");
plt.show()

plt.plot(fpr_test, tpr_test, color='orange',
         lw=1.5, label='AUC = %0.2f' % auc_test)
plt.plot([0, 1], [0, 1], color='blue', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Baseline Test')
plt.legend(loc="lower right");

#confusion matrix for validation
conf_val = confusion_matrix(y_val, pred_tuneval.round())
conf_test = confusion_matrix(y_test, pred_tunetest.round())

plt.figure(figsize=(12,6))
plt.subplot(121)
sn.heatmap(conf_val, annot = True, fmt=".3f", square=True, cmap = "Blues_r");
plt.ylabel('True label');
plt.xlabel('Predicted label')
plt.title('Confusion matrix - Baseline Validation')
plt.subplot(122)
sn.heatmap(conf_test, annot = True, fmt=".3f", square = True, cmap = "Blues_r");
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix - Baseline Test')
plt.savefig('Confusion matrix - Baseline Validation and Test')
plt.show()

metrics = model_rus.evaluate(X_test_scaled, y_test)
metrics


# ### Tuning Random oversampler

# In[95]:


seed(42)
tf.random.set_seed(42)
## tuner for unsampled training set
tuner_ros = kt.BayesianOptimization(
    build_model,
    objective=kt.Objective("auc", direction="max"),
    directory='directory',
    project_name='Tuned_parameters',
    max_trials=5,
    overwrite=True)


# In[96]:


seed(42)
tf.random.set_seed(42)

tuner_ros.search(X_train_ros_scaled, y_ros, epochs=20, validation_data=(X_val_scaled, y_val))
best_model3 = tuner_ros.get_best_models()[0]


# In[97]:


seed(42)
tf.random.set_seed(42)

ros_models = tuner_ros.get_best_models(num_models=3)
best_model_ros = ros_models[0]

best_model_ros.build(input_shape=(39820, 69))
best_model_ros.summary()

best_hps_ros = tuner_ros.get_best_hyperparameters(5)
# Build the model with the best hp.
model_ros = build_model(best_hps_ros[0])

ros_tuned = model_ros.fit(X_train_ros_scaled, y_ros, batch_size= 32, epochs = 50, 
          validation_data = (X_val_scaled, y_val))


# In[98]:


plt.style.use('seaborn')
plt.figure(figsize = (16,5))
plt.subplot(121)
plt.title('loss with epochs', fontsize=16)
plt.plot(ros_tuned.history['loss'], marker='o', color="orange", label="Training loss")
plt.plot(ros_tuned.history['val_loss'], marker='o',color="blue", label="Validation loss")
plt.xticks()
plt.legend()

plt.subplot(122)
plt.title('accuracy with epochs', fontsize=16)
plt.plot(ros_tuned.history['accuracy'], marker='o', color="orange", label="Training Acc")
plt.plot(ros_tuned.history['val_accuracy'], marker='o',color="blue", label="Validation Acc")
plt.xticks()
plt.legend()


# In[130]:


pred_tuneval = model_ros.predict(X_val_scaled, verbose = 1).ravel()
pred_tunetest = model_ros.predict(X_test_scaled, verbose = 1).ravel()

fpr_val, tpr_val, thresholds_val = roc_curve(y_val, pred_tuneval)
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, pred_tunetest)

auc_val = auc(fpr_val, tpr_val)
auc_test = auc(fpr_test, tpr_test)

plt.plot(fpr_val, tpr_val, color='orange',
         lw=1.5, label='AUC = %0.2f' % auc_val)
plt.plot([0, 1], [0, 1], color='blue', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Baseline Validation')
plt.legend(loc="lower right");
plt.show()

plt.plot(fpr_test, tpr_test, color='orange',
         lw=1.5, label='AUC = %0.2f' % auc_test)
plt.plot([0, 1], [0, 1], color='blue', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Baseline Test')
plt.legend(loc="lower right");

#confusion matrix for validation
conf_val = confusion_matrix(y_val, pred_tuneval.round())
conf_test = confusion_matrix(y_test, pred_tunetest.round())

plt.figure(figsize=(12,6))
plt.subplot(121)
sn.heatmap(conf_val, annot = True, fmt=".3f", square=True, cmap = "Blues_r");
plt.ylabel('True label');
plt.xlabel('Predicted label')
plt.title('Confusion matrix - Baseline Validation')
plt.subplot(122)
sn.heatmap(conf_test, annot = True, fmt=".3f", square = True, cmap = "Blues_r");
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix - Tuned oversampled Test')
plt.savefig('Confusion matrix - Baseline Validation and Test')
plt.show()

metrics = model_ros.evaluate(X_test_scaled, y_test)
metrics


# ### Tuning SMOTE

# In[100]:


seed(42)
tf.random.set_seed(42)
## tuner for unsampled training set
tuner_smote = kt.BayesianOptimization(
    build_model,
    objective=kt.Objective("auc", direction="max"),
    directory='directory',
    project_name='Tuned_parameters',
    max_trials=5,
    overwrite=True)


# In[101]:


seed(42)
tf.random.set_seed(42)

tuner_smote.search(X_train_smote_scaled, y_smote, epochs=20, validation_data=(X_val_scaled, y_val))
best_model4 = tuner_smote.get_best_models()[0]


# In[102]:


seed(42)
tf.random.set_seed(42)

smote_models = tuner_smote.get_best_models(num_models=3)
best_model_smote = smote_models[0]

best_model_smote.build(input_shape=(39820, 69))
best_model_smote.summary()

best_hps_smote = tuner_smote.get_best_hyperparameters(5)
# Build the model with the best hp.
model_smote = build_model(best_hps_smote[0])

smote_tuned = model_smote.fit(X_train_smote_scaled, y_smote, batch_size= 32, epochs = 50, 
          validation_data = (X_val_scaled, y_val))


# In[103]:


plt.style.use('seaborn')
plt.figure(figsize = (16,5))
plt.subplot(121)
plt.title('loss with epochs', fontsize=16)
plt.plot(smote_tuned.history['loss'], marker='o', color="orange", label="Training loss")
plt.plot(smote_tuned.history['val_loss'], marker='o',color="blue", label="Validation loss")
plt.xticks()
plt.legend()

plt.subplot(122)
plt.title('accuracy with epochs', fontsize=16)
plt.plot(smote_tuned.history['accuracy'], marker='o', color="orange", label="Training Acc")
plt.plot(smote_tuned.history['val_accuracy'], marker='o',color="blue", label="Validation Acc")
plt.xticks()
plt.legend()


# In[104]:


pred_tuneval = model_smote.predict(X_val_scaled, verbose = 1).ravel()
pred_tunetest = model_smote.predict(X_test_scaled, verbose = 1).ravel()

fpr_val, tpr_val, thresholds_val = roc_curve(y_val, pred_tuneval)
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, pred_tunetest)

auc_val = auc(fpr_val, tpr_val)
auc_test = auc(fpr_test, tpr_test)

plt.plot(fpr_val, tpr_val, color='orange',
         lw=1.5, label='AUC = %0.2f' % auc_val)
plt.plot([0, 1], [0, 1], color='blue', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Baseline Validation')
plt.legend(loc="lower right");
plt.show()

plt.plot(fpr_test, tpr_test, color='orange',
         lw=1.5, label='AUC = %0.2f' % auc_test)
plt.plot([0, 1], [0, 1], color='blue', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Baseline Test')
plt.legend(loc="lower right");

#confusion matrix for validation
conf_val = confusion_matrix(y_val, pred_tuneval.round())
conf_test = confusion_matrix(y_test, pred_tunetest.round())

plt.figure(figsize=(12,6))
plt.subplot(121)
sn.heatmap(conf_val, annot = True, fmt=".3f", square=True, cmap = "Blues_r");
plt.ylabel('True label');
plt.xlabel('Predicted label')
plt.title('Confusion matrix - Baseline Validation')
plt.subplot(122)
sn.heatmap(conf_test, annot = True, fmt=".3f", square = True, cmap = "Blues_r");
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix - Baseline Test')
plt.savefig('Confusion matrix - Baseline Validation and Test')
plt.show()

metrics = model_smote.evaluate(X_test_scaled, y_test)
metrics


# In[ ]:





# ## Tuning models with mutual information feature selection

# ### Tuning baseline with mutual information feature selection

# In[105]:


seed(42)
tf.random.set_seed(42)
## tuner for unsampled training set with mutual information feature selection
tuner_baseline_mi = kt.BayesianOptimization(
    build_model,
    objective=kt.Objective("auc", direction="max"),
    directory='directory',
    project_name='Tuned_parameters',
    max_trials=5,
    overwrite=True)


# In[106]:


seed(42)
tf.random.set_seed(42)

tuner_baseline_mi.search(X_train_mi, y_train, epochs=20, validation_data=(X_val_mi, y_val))
best_model1_mi = tuner_baseline_mi.get_best_models()[0]


# In[107]:


# tuner_baseline.results_summary()


# In[108]:


seed(42)
tf.random.set_seed(42)

base_models_mi = tuner_baseline_mi.get_best_models(num_models=3)
best_model_base_mi = base_models_mi[0]

best_model_base_mi.build(input_shape=(39820, 20))
best_model_base_mi.summary()

best_hps_base_mi = tuner_baseline_mi.get_best_hyperparameters(5)
# Build the model with the best hp.
model_base_mi = build_model(best_hps_base_mi[0])

baseline_tuned_mi = model_base_mi.fit(X_train_mi, y_train, batch_size= 32, epochs = 50, 
          validation_data = (X_val_mi, y_val))


# In[109]:


plt.style.use('seaborn')
plt.figure(figsize = (16,5))
plt.subplot(121)
plt.title('loss with epochs', fontsize=16)
plt.plot(baseline_tuned_mi.history['loss'], marker='o', color="orange", label="Training loss")
plt.plot(baseline_tuned_mi.history['val_loss'], marker='o',color="blue", label="Validation loss")
plt.xticks()
plt.legend()

plt.subplot(122)
plt.title('accuracy with epochs', fontsize=16)
plt.plot(baseline_tuned_mi.history['accuracy'], marker='o', color="orange", label="Training Acc")
plt.plot(baseline_tuned_mi.history['val_accuracy'], marker='o',color="blue", label="Validation Acc")
plt.xticks()
plt.legend()


# In[110]:


pred_tuneval_mi = model_base_mi.predict(X_val_mi, verbose = 1).ravel()
pred_tunetest_mi = model_base_mi.predict(X_test_mi, verbose = 1).ravel()

fpr_val, tpr_val, thresholds_val = roc_curve(y_val, pred_tuneval_mi)
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, pred_tunetest_mi)

auc_val = auc(fpr_val, tpr_val)
auc_test = auc(fpr_test, tpr_test)

plt.plot(fpr_val, tpr_val, color='orange',
         lw=1.5, label='AUC = %0.2f' % auc_val)
plt.plot([0, 1], [0, 1], color='blue', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Baseline Validation')
plt.legend(loc="lower right");
plt.show()

plt.plot(fpr_test, tpr_test, color='orange',
         lw=1.5, label='AUC = %0.2f' % auc_test)
plt.plot([0, 1], [0, 1], color='blue', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Baseline Test')
plt.legend(loc="lower right");

#confusion matrix for validation
conf_val = confusion_matrix(y_val, pred_tuneval_mi.round())
conf_test = confusion_matrix(y_test, pred_tunetest_mi.round())

plt.figure(figsize=(12,6))
plt.subplot(121)
sn.heatmap(conf_val, annot = True, fmt=".3f", square=True, cmap = "Blues_r");
plt.ylabel('True label');
plt.xlabel('Predicted label')
plt.title('Confusion matrix - Baseline Validation')
plt.subplot(122)
sn.heatmap(conf_test, annot = True, fmt=".3f", square = True, cmap = "Blues_r");
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix - Baseline Test')
plt.savefig('Confusion matrix - Baseline Validation and Test')
plt.show()

metrics = model_base_mi.evaluate(X_test_mi, y_test)
metrics


# ## Tuning undersampled model with mutual information feature selection

# In[111]:


seed(42)
tf.random.set_seed(42)
## tuner for undersampled training set with mutual inforamtion feature selection
tuner_rus_mi = kt.BayesianOptimization(
    build_model,
    objective=kt.Objective("auc", direction="max"),
    directory='directory',
    project_name='Tuned_parameters',
    max_trials=5,
    overwrite=True)


# In[112]:


seed(42)
tf.random.set_seed(42)

tuner_rus_mi.search(X_train_mi, y_train, epochs=20, validation_data=(X_val_mi, y_val))
best_model2_mi = tuner_rus_mi.get_best_models()[0]


# In[113]:


# tuner_baseline.results_summary()


# In[114]:


seed(42)
tf.random.set_seed(42)

rus_models_mi = tuner_rus_mi.get_best_models(num_models=3)
best_model_rus_mi = rus_models_mi[0]

best_model_rus_mi.build(input_shape=(39820, 20))
best_model_rus_mi.summary()

best_hps_rus_mi = tuner_rus_mi.get_best_hyperparameters(5)
# Build the model with the best hp.
model_rus_mi = build_model(best_hps_rus_mi[0])

rus_tuned_mi = model_rus_mi.fit(X_train_mi, y_train, batch_size= 32, epochs = 50, 
          validation_data = (X_val_mi, y_val))


# In[115]:


plt.style.use('seaborn')
plt.figure(figsize = (16,5))
plt.subplot(121)
plt.title('loss with epochs', fontsize=16)
plt.plot(rus_tuned_mi.history['loss'], marker='o', color="orange", label="Training loss")
plt.plot(rus_tuned_mi.history['val_loss'], marker='o',color="blue", label="Validation loss")
plt.xticks()
plt.legend()

plt.subplot(122)
plt.title('accuracy with epochs', fontsize=16)
plt.plot(rus_tuned_mi.history['accuracy'], marker='o', color="orange", label="Training Acc")
plt.plot(rus_tuned_mi.history['val_accuracy'], marker='o',color="blue", label="Validation Acc")
plt.xticks()
plt.legend()


# In[116]:


pred_tuneval_mi = model_rus_mi.predict(X_val_mi, verbose = 1).ravel()
pred_tunetest_mi = model_rus_mi.predict(X_test_mi, verbose = 1).ravel()

fpr_val, tpr_val, thresholds_val = roc_curve(y_val, pred_tuneval_mi)
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, pred_tunetest_mi)

auc_val = auc(fpr_val, tpr_val)
auc_test = auc(fpr_test, tpr_test)

plt.plot(fpr_val, tpr_val, color='orange',
         lw=1.5, label='AUC = %0.2f' % auc_val)
plt.plot([0, 1], [0, 1], color='blue', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Baseline Validation')
plt.legend(loc="lower right");
plt.show()

plt.plot(fpr_test, tpr_test, color='orange',
         lw=1.5, label='AUC = %0.2f' % auc_test)
plt.plot([0, 1], [0, 1], color='blue', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Baseline Test')
plt.legend(loc="lower right");

#confusion matrix for validation
conf_val = confusion_matrix(y_val, pred_tuneval_mi.round())
conf_test = confusion_matrix(y_test, pred_tunetest_mi.round())

plt.figure(figsize=(12,6))
plt.subplot(121)
sn.heatmap(conf_val, annot = True, fmt=".3f", square=True, cmap = "Blues_r");
plt.ylabel('True label');
plt.xlabel('Predicted label')
plt.title('Confusion matrix - Baseline Validation')
plt.subplot(122)
sn.heatmap(conf_test, annot = True, fmt=".3f", square = True, cmap = "Blues_r");
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix - Baseline Test')
plt.savefig('Confusion matrix - Baseline Validation and Test')
plt.show()

metrics = model_rus_mi.evaluate(X_test_mi, y_test)
metrics


# In[ ]:





# ## Tuning oversampled model with mutual information feature selection

# In[117]:


seed(42)
tf.random.set_seed(42)
## tuner for oversampled training set with mutual information feature selection
tuner_ros_mi = kt.BayesianOptimization(
    build_model,
    objective=kt.Objective("auc", direction="max"),
    directory='directory',
    project_name='Tuned_parameters',
    max_trials=5,
    overwrite=True)


# In[118]:


seed(42)
tf.random.set_seed(42)

tuner_ros_mi.search(X_train_mi, y_train, epochs=20, validation_data=(X_val_mi, y_val))
best_model3_mi = tuner_ros_mi.get_best_models()[0]


# In[119]:


# tuner_baseline.results_summary()


# In[120]:


seed(42)
tf.random.set_seed(42)

ros_models_mi = tuner_ros_mi.get_best_models(num_models=3)
best_model_ros_mi = ros_models_mi[0]

best_model_ros_mi.build(input_shape=(39820, 20))
best_model_ros_mi.summary()

best_hps_ros_mi = tuner_ros_mi.get_best_hyperparameters(5)
# Build the model with the best hp.
model_ros_mi = build_model(best_hps_ros_mi[0])

ros_tuned_mi = model_ros_mi.fit(X_train_mi, y_train, batch_size= 32, epochs = 50, 
          validation_data = (X_val_mi, y_val))


# In[121]:


plt.style.use('seaborn')
plt.figure(figsize = (16,5))
plt.subplot(121)
plt.title('loss with epochs', fontsize=16)
plt.plot(ros_tuned_mi.history['loss'], marker='o', color="orange", label="Training loss")
plt.plot(ros_tuned_mi.history['val_loss'], marker='o',color="blue", label="Validation loss")
plt.xticks()
plt.legend()

plt.subplot(122)
plt.title('accuracy with epochs', fontsize=16)
plt.plot(ros_tuned_mi.history['accuracy'], marker='o', color="orange", label="Training Acc")
plt.plot(ros_tuned_mi.history['val_accuracy'], marker='o',color="blue", label="Validation Acc")
plt.xticks()
plt.legend()


# In[122]:


pred_tuneval_mi = model_ros_mi.predict(X_val_mi, verbose = 1).ravel()
pred_tunetest_mi = model_ros_mi.predict(X_test_mi, verbose = 1).ravel()

fpr_val, tpr_val, thresholds_val = roc_curve(y_val, pred_tuneval_mi)
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, pred_tunetest_mi)

auc_val = auc(fpr_val, tpr_val)
auc_test = auc(fpr_test, tpr_test)

plt.plot(fpr_val, tpr_val, color='orange',
         lw=1.5, label='AUC = %0.2f' % auc_val)
plt.plot([0, 1], [0, 1], color='blue', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Baseline Validation')
plt.legend(loc="lower right");
plt.show()

plt.plot(fpr_test, tpr_test, color='orange',
         lw=1.5, label='AUC = %0.2f' % auc_test)
plt.plot([0, 1], [0, 1], color='blue', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Baseline Test')
plt.legend(loc="lower right");

#confusion matrix for validation
conf_val = confusion_matrix(y_val, pred_tuneval_mi.round())
conf_test = confusion_matrix(y_test, pred_tunetest_mi.round())

plt.figure(figsize=(12,6))
plt.subplot(121)
sn.heatmap(conf_val, annot = True, fmt=".3f", square=True, cmap = "Blues_r");
plt.ylabel('True label');
plt.xlabel('Predicted label')
plt.title('Confusion matrix - Baseline Validation')
plt.subplot(122)
sn.heatmap(conf_test, annot = True, fmt=".3f", square = True, cmap = "Blues_r");
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix - Baseline Test')
plt.savefig('Confusion matrix - Baseline Validation and Test')
plt.show()

metrics = model_ros_mi.evaluate(X_test_mi, y_test)
metrics


# In[ ]:





# ## Tuning SMOTE model with mutual information feature selection

# In[123]:


seed(42)
tf.random.set_seed(42)
## tuner for undersampled training set with mutual inforamtion feature selection
tuner_smote_mi = kt.BayesianOptimization(
    build_model,
    objective=kt.Objective("auc", direction="max"),
    directory='directory',
    project_name='Tuned_parameters',
    max_trials=5,
    overwrite=True)


# In[124]:


seed(42)
tf.random.set_seed(42)

tuner_smote_mi.search(X_train_mi, y_train, epochs=20, validation_data=(X_val_mi, y_val))
best_model4_mi = tuner_smote_mi.get_best_models()[0]


# In[125]:


# tuner_baseline.results_summary()


# In[126]:


seed(42)
tf.random.set_seed(42)

smote_models_mi = tuner_smote_mi.get_best_models(num_models=3)
best_model_smote_mi = smote_models_mi[0]

best_model_smote_mi.build(input_shape=(39820, 20))
best_model_smote_mi.summary()

best_hps_smote_mi = tuner_smote_mi.get_best_hyperparameters(5)
# Build the model with the best hp.
model_smote_mi = build_model(best_hps_smote_mi[0])

smote_tuned_mi = model_smote_mi.fit(X_train_mi, y_train, batch_size= 32, epochs = 50, 
          validation_data = (X_val_mi, y_val))


# In[127]:


plt.style.use('seaborn')
plt.figure(figsize = (16,5))
plt.subplot(121)
plt.title('loss with epochs', fontsize=16)
plt.plot(smote_tuned_mi.history['loss'], marker='o', color="orange", label="Training loss")
plt.plot(smote_tuned_mi.history['val_loss'], marker='o',color="blue", label="Validation loss")
plt.xticks()
plt.legend()

plt.subplot(122)
plt.title('accuracy with epochs', fontsize=16)
plt.plot(smote_tuned_mi.history['accuracy'], marker='o', color="orange", label="Training Acc")
plt.plot(smote_tuned_mi.history['val_accuracy'], marker='o',color="blue", label="Validation Acc")
plt.xticks()
plt.legend()


# In[128]:


pred_tuneval_mi = model_smote_mi.predict(X_val_mi, verbose = 1).ravel()
pred_tunetest_mi = model_smote_mi.predict(X_test_mi, verbose = 1).ravel()

fpr_val, tpr_val, thresholds_val = roc_curve(y_val, pred_tuneval_mi)
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, pred_tunetest_mi)

auc_val = auc(fpr_val, tpr_val)
auc_test = auc(fpr_test, tpr_test)

plt.plot(fpr_val, tpr_val, color='orange',
         lw=1.5, label='AUC = %0.2f' % auc_val)
plt.plot([0, 1], [0, 1], color='blue', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Baseline Validation')
plt.legend(loc="lower right");
plt.show()

plt.plot(fpr_test, tpr_test, color='orange',
         lw=1.5, label='AUC = %0.2f' % auc_test)
plt.plot([0, 1], [0, 1], color='blue', lw=1.5, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Baseline Test')
plt.legend(loc="lower right");

#confusion matrix for validation
conf_val = confusion_matrix(y_val, pred_tuneval_mi.round())
conf_test = confusion_matrix(y_test, pred_tunetest_mi.round())

plt.figure(figsize=(12,6))
plt.subplot(121)
sn.heatmap(conf_val, annot = True, fmt=".3f", square=True, cmap = "Blues_r");
plt.ylabel('True label');
plt.xlabel('Predicted label')
plt.title('Confusion matrix - Baseline Validation')
plt.subplot(122)
sn.heatmap(conf_test, annot = True, fmt=".3f", square = True, cmap = "Blues_r");
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix - Baseline Test')
plt.savefig('Confusion matrix - Baseline Validation and Test')
plt.show()

metrics = model_smote_mi.evaluate(X_test_mi, y_test)
metrics

