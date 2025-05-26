#!/usr/bin/env python
# coding: utf-8




import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score


# In[28]:


path=r'C:\Users\ADMIN\Desktop\AYVID Training\AI & ML\DATASETS\breast cancer.csv'


# In[29]:


df=pd.read_csv(path)


# In[30]:


df.head(15)


# In[31]:


df.isnull().sum()


# In[32]:


# dropping uneccesary columns
df.drop(['id','Unnamed: 32'], axis = 1, inplace = True)


# In[33]:


df.info()


# In[34]:


df.describe()


# In[41]:


df['diagnosis'] = df['diagnosis'].replace({'M': 1, 'B': 0})


# In[ ]:





# In[42]:


df['diagnosis'].value_counts()


# In[43]:


#####balanced dataset


# In[44]:


x = df.drop('diagnosis', axis = 1)
y = df['diagnosis']


# In[45]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state = 42)


# In[46]:


y_train.value_counts()


# In[47]:


y_test.value_counts()


# In[48]:


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# In[50]:


model = LogisticRegression(max_iter=1000)
model.fit(x_train_scaled, y_train)


# In[55]:


y_pred = model.predict(x_test_scaled)
acc= accuracy_score(y_test, y_pred)


# In[56]:


acc


# In[85]:


cr=classification_report(y_test, y_pred)


# In[86]:


print(cr)


# In[87]:


roc_auc_score(y_test, y_pred)


# In[61]:


y_pred


# In[63]:


prob = model.predict_proba(x_test_scaled)
prob


# In[64]:


dummy = prob[0:10] ####takes the first 10 predictions (as probability pairs)


# In[65]:


dummy


# In[70]:


dummy[3][0]  ##probability of class 0 for the 3rd sample


# In[71]:


threshold = 0.8
new_class = []
for i in range(len(dummy)):
    if dummy[i][0] > threshold:
        new_class.append(0)
    else:
        new_class.append(1)


# In[75]:


new=np.array(new_class)
new


# In[81]:


cr1=classification_report(y_test[0:10], new)


# In[82]:


print(cr1)


# In[83]:


rc1=roc_auc_score(y_test[0:10], new)


# In[84]:


rc1


# In[88]:


print(cr)


# In[ ]:




