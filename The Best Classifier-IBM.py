#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[276]:


data=pd.read_csv('loan_train.csv')


# In[277]:


data.head(5)


# In[278]:


data.describe()


# In[279]:


data.info()


# In[280]:


data.shape


# In[281]:


data.columns


# In[282]:


data['due_date']=pd.to_datetime(data['due_date'])
data['effective_date']=pd.to_datetime(data['effective_date'])
data.head(2)


# In[283]:


data['loan_status'].value_counts()


# In[284]:


data.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# In[285]:


data.groupby(['education'])['loan_status'].value_counts()


# In[286]:


data['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
data.head()


# In[287]:


data['dayofweek'] = data['effective_date'].dt.dayofweek
data.head()


# In[288]:


data['weekend'] = data['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
data.head()


# In[289]:


cat_feats=['education']


# In[290]:


final_df=pd.get_dummies(data,columns=cat_feats)


# In[291]:


final_df.head(2)


# In[292]:


df=final_df.drop('education_Master or Above',axis=1)


# In[293]:


df.head()


# In[294]:


df.columns


# In[295]:


df=df.drop(['Unnamed: 0', 'Unnamed: 0.1','effective_date', 'due_date', 'dayofweek'],axis=1)


# In[296]:


df.head()


# In[297]:


X=df.drop('loan_status',axis=1).values
y=df['loan_status'].values


# In[298]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[299]:


X=scaler.fit(X).transform(X)


# In[300]:


from sklearn.model_selection import train_test_split


# In[301]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[ ]:





# In[302]:


print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# # KNeighbors Classifier

# In[303]:


from sklearn.neighbors import KNeighborsClassifier


# In[304]:


#Train Model and Predict

knn = KNeighborsClassifier(n_neighbors = 1)


# In[305]:


knn.fit(X_train,y_train)


# In[306]:


pred=knn.predict(X_test)


# In[307]:


from sklearn import metrics


# In[308]:


from sklearn.metrics import confusion_matrix,classification_report


# In[309]:


print(confusion_matrix(y_test,pred))


# In[310]:


print(classification_report(y_test,pred))


# #
# 

# In[311]:


error_rate=[]

for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i=knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    


# In[312]:


plt.plot(range(1,40),error_rate,color='blue')


# In[313]:


#for better result use k=19
knn=KNeighborsClassifier(n_neighbors=19)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# # Decision Tree

# In[314]:


from sklearn.tree import DecisionTreeClassifier


# In[366]:


dtree=DecisionTreeClassifier(criterion='entropy',max_depth=9)


# In[367]:


dtree.fit(X_train,y_train)


# In[368]:


D_pred=dtree.predict(X_test)


# In[369]:


print(confusion_matrix(y_test,D_pred))


# In[370]:


print(classification_report(y_test,D_pred))


# # SVM

# In[371]:


from sklearn.svm import SVC


# In[372]:


svc_model=SVC()


# In[373]:


svc_model.fit(X_train,y_train)


# In[374]:


svc_pred=svc_model.predict(X_test)


# In[376]:


print(confusion_matrix(y_test,svc_pred))


# In[377]:


print(classification_report(y_test,svc_pred))


# In[379]:


# for better performance
from sklearn.model_selection import GridSearchCV


# In[380]:


param_grid={'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}


# In[381]:


grid=GridSearchCV(SVC(),param_grid,verbose=2)


# In[382]:


grid.fit(X_train,y_train)


# In[383]:


grid.best_params_


# In[385]:


final_svc_pred=grid.predict(X_test)


# In[386]:


print(classification_report(y_test,final_svc_pred))


# In[387]:


print(confusion_matrix(y_test,final_svc_pred))


# # Logistic Regression

# In[389]:


from sklearn.linear_model import LogisticRegression


# In[390]:


lr_model=LogisticRegression()


# In[391]:


lr_model.fit(X_train,y_train)


# In[393]:


lr_pred=lr_model.predict(X_test)


# In[394]:


print(classification_report(y_test,lr_pred))


# In[395]:


print(confusion_matrix(y_test,lr_pred))


# # Model Evaluation by using test set

# In[396]:


from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# In[441]:


test=pd.read_csv('loan_test.csv')


# In[442]:


test.head(2)


# In[443]:


test['due_date']=pd.to_datetime(test['due_date'])
test['effective_date']=pd.to_datetime(test['effective_date'])

test['dayofweek']=test['effective_date'].dt.dayofweek
test['weekend']=test['dayofweek'].apply(lambda x:1 if (x>3) else 0)
test.head()


# In[444]:


cat_feats=['education']
test_df=pd.get_dummies(test,columns=cat_feats)

test_df=test_df.drop('education_Master or Above',axis=1)
test_df=test_df.drop(['Unnamed: 0', 'Unnamed: 0.1','effective_date', 'due_date', 'dayofweek'],axis=1)


# In[445]:


test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
test_df.head()


# In[470]:


test_X=test_df.drop('loan_status',axis=1).values
test_y=test_df['loan_status'].values


# In[471]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[472]:


test_X=scaler.fit(X).transform(X)


# In[473]:


#KNN Evaluation
test_knn_pred = knn.predict(test_X)
print("KNN Jaccard index: %.2f " % jaccard_score(test_y, test_knn_pred,pos_label= 'PAIDOFF'))
print("KNN F1-score: %.2f " % f1_score(test_y, test_knn_pred,pos_label='PAIDOFF'))


# In[474]:


#Decision Tree Evaluation
test_dtree_pred = dtree.predict(test_X)
print("dtree Jaccard index: %.2f " % jaccard_score(test_y, test_dtree_pred,pos_label= 'PAIDOFF'))
print("dtree F1-score: %.2f " % f1_score(test_y, test_dtree_pred,pos_label='PAIDOFF'))


# In[475]:


#SVM Evaluation
test_svc_pred = svc_model.predict(test_X)
print("SVM Jaccard index: %.2f " % jaccard_score(test_y, test_svc_pred,pos_label= 'PAIDOFF'))
print("SVM F1-score: %.2f " %  f1_score(test_y, test_svc_pred,pos_label='PAIDOFF'))


# In[476]:


#Logistics Regression Evaluation
test_lr_pred = lr_model.predict(test_X)
test_lr_pred_prob = lr_model.predict_proba(test_X)
print("LR Jaccard index: %.2f" % jaccard_score(test_y, test_lr_pred, pos_label='PAIDOFF'))
print("LR F1-score: %.2f" % f1_score(test_y, test_lr_pred, pos_label='PAIDOFF') )
print("LR LogLoss: %.2f" % log_loss(test_y, test_lr_pred_prob))


# In[ ]:




