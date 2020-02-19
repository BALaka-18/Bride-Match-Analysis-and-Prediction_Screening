#!/usr/bin/env python
# coding: utf-8

# ## Creating a model on the prediction of match/not a match based on Indian dataset of brides taken from a matrimonial portal.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as ml
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
ml.style.use('ggplot')


# In[2]:


m = pd.read_csv('data.csv')
m.head(30)


# In[3]:


m.shape


# In[4]:


m.info()


# ## Cleaning the dataset to get better insights

# In[5]:


m.drop(['about1','about2','about_family'], axis=1, inplace=True)
m


# In[6]:


m[m['body1'].isnull() == True]


# In[7]:


m['body1'].value_counts()


# In[8]:


m.groupby('body1')['body1'].agg('count').plot(kind='bar')
plt.xlabel("Body Type")
plt.ylabel("Count")
plt.title("Most popular Body type")
plt.show()


# Most women have Average body type

# In[9]:


m['body1'].replace('Heavy',1,inplace=True)
m[m['body1'] == 1]


# In[10]:


m['body1'].replace('Average',2,inplace=True)
m['body1'].replace('Slim',3,inplace=True)
m['body1'].replace('Athletic',4,inplace=True)
m['body1'].replace('Doesn\'t Matter',np.nan,inplace=True)
m['body1'].fillna(0,inplace=True)
m.drop(columns=['body2'], inplace=True)
m


# In[11]:


m['caste1'].value_counts()


# In[12]:


m[m['caste1'].isnull() == True]
m['caste1'].fillna('Not Provided',inplace=True)
m


# In[13]:


m.groupby('caste1')['caste1'].agg('count').sort_values(ascending=False).head(20).plot(kind='barh',color='Teal')
plt.ylabel("Caste")
plt.xlabel("Count")
plt.title("Caste Count (Top 20)")
plt.show()


# Most women belong to the Brahmin caste

# In[ ]:


from sklearn.preprocessing import LabelEncoder          # Encoding the textual columns

le = LabelEncoder()
m['new_caste'] = le.fit_transform(m['caste1'])


# In[15]:


m.drop(columns=['caste1','caste2'],inplace=True)
m


# In[16]:


m['complexion1'].value_counts()
m['complexion1'].count()

m['complexion1'].replace({'Fair':'Fair ',
                           'Wheatish':'Wheatish ',
                         'Whetish Medium':'Whetish Medium ',
                          'Dark':'Dark '},inplace=True)
m['complexion1'].fillna('Doesn\'t Matter',inplace=True)
print(m['complexion1'].value_counts())
m.groupby('complexion1')['complexion1'].agg('count').sort_values(ascending=True).plot(kind='bar',color='green')
plt.xlabel("Complexion")
plt.ylabel("Count")
plt.title("Complexion Count")
plt.show()


# Most women in the matrimonial portal are Fair skinned

# In[17]:


m['complexion'] = le.fit_transform(m['complexion1'])
m['complexion'].value_counts()


# In[18]:


m['values1'].value_counts()


# In[19]:


m['values2'].value_counts()


# In[20]:


m['values2'].fillna('Match',inplace=True)
m['values1'].fillna('Not Provided',inplace=True)
m


# In[21]:


m['values2'].value_counts()


# In[22]:


m[m['weight1'].isnull() == True]
m['weight1'].fillna(0,inplace=True)
m


# In[23]:


m['weight1'] = m['weight1'].astype('str')
m['wt'] = m['weight1'].str.slice(0,3)
m['wt'].replace('0 t',51,inplace=True)
m['wt'].replace('0',51,inplace=True)
m['wt'].replace('mor',51,inplace=True)
m['wt'].replace('les',51,inplace=True)

m['wt'].value_counts()
m['wt'] = m['wt'].astype('int')
m


# In[24]:


m.drop(columns=['weight1','weight2'],inplace=True)
m


# In[25]:


m['smoking1'].value_counts()


# In[26]:


m['smoking1'].fillna('No',inplace=True)
m.drop(columns=['smoking2'],inplace=True)
m


# In[27]:


m['smoke'] = le.fit_transform(m['smoking1'])
m


# In[28]:


m['have_children'] = m['have_children'].astype('str')
m['child'] = le.fit_transform(m['have_children'])
m[m['child'] == 1]


# In[29]:


m['drinking1'].value_counts()
m['drinking1'].fillna('No',inplace=True)
m['drinking1'].value_counts()


# In[33]:


m['employed1'].value_counts()


# In[34]:


m['employed1'].fillna('Not Working',inplace=True)
m['employed1'].value_counts()


# In[35]:


m['employed'] = le.fit_transform(m['employed1'])
m['employed'].value_counts()


# In[36]:


m.drop(columns=['employed1','employed2'], inplace=True)
m


# In[37]:


m['religion1'].value_counts()


# In[38]:


m['religion1'].fillna('Others',inplace=True)
m['religion'] = le.fit_transform(m['religion1'])
m['religion'].value_counts()


# In[39]:


m.drop(columns=['religion1','religion2'],inplace=True)
m


# In[40]:


m


# In[41]:


m['height1'] = m['height1'].astype('str')
m['ht_ft'] = m['height1'].str.slice(0,1)
m['ht_ft'].replace('n','0',inplace=True)
m['ht_ft'] = m['ht_ft'].astype('int')
m['ht_in'] = m['height1'].str.slice(4,5)
m['ht_in'].replace('','0',inplace=True)
m['ht_in'] = m['ht_in'].astype('int')
m['ht'] = m['ht_ft'] + (0.1*m['ht_in'])
m.drop(columns=['height1','height2','ht_ft','ht_in'],inplace=True)


# In[42]:


# m.drop(columns=['city2','complexion1','complexion2'],inplace=True)
m.drop(columns=['age2','birth1','birth2','birth3','birth4','brothers','college','created_for','degree'],inplace=True)
m


# In[43]:


m.drop(columns=['details','education2','family_type1','family_type2'],inplace=True)
m


# In[44]:


m['drinking1'].value_counts()


# In[45]:


m['drinking'] = le.fit_transform(m['drinking1'])
m['drinking'].value_counts()


# In[46]:


m.drop(columns=['drinking1','drinking2','fields','gotra','have_children'],inplace=True)
m


# In[47]:


m['eating1'].value_counts()

m['eating1'] = m['eating1'].astype('str')
m['eat'] = le.fit_transform(m['eating1'])
m['eat'].value_counts()


# In[48]:


m.drop(columns=['eating1','eating2','father','horoscope','income1','income2','manglik1','manglik2'],inplace=True)
m


# In[49]:


m.drop(columns=['drinking'],inplace=True)
m


# In[50]:


m['marital_status1'].value_counts()
m['marital_status1'].fillna('Never Married',inplace=True)

m['marital_status'] = le.fit_transform(m['marital_status1'])
m['marital_status'].value_counts()
m.drop(columns=['marital_status1','marital_status2','mother','mother1','mother2','nakshatra','occupation'],inplace=True)
m


# In[51]:


m.drop(columns=['raasi','sisters','smoking1','last_login','living'],inplace=True)
m


# In[52]:


m.drop(columns=['special1','special2','status2','subcaste','url'],inplace=True)
m


# In[53]:


m['status1'].value_counts()
m['status1'].fillna('Doesn\'t Matter',inplace=True)
m['status1'].value_counts()


# In[54]:


m['stat'] = le.fit_transform(m['status1'])
m['stat'].value_counts()


# In[55]:


m.drop(columns=['status1'],inplace=True)
m


# In[56]:


m['values_wanted'] = le.fit_transform(m['values1'])
print(m['values1'].value_counts())
print(m['values_wanted'].value_counts())


# In[57]:


m.drop(columns=['values1'],inplace=True)
m


# After removing all unwanted columns and cleaning the dataset to get rid of misfit data, we get a ready-to-work-on dataset that has been reduced from 66 columns to 18 columns

# In[58]:


m['values_matched'] = le.fit_transform(m['values2'])
m.drop(columns=['values2'],inplace=True)
m


# In[59]:


m['values_matched'].value_counts()


# ## Defining features and labels
# 
# Extracting required columns for feature matrix and label Series. Then splitting into training and testing data.

# In[85]:


X = m.iloc[:,[0,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2]]
X


# In[86]:


Y = m.iloc[:,-1]
Y


# In[87]:


from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.2)


# In[88]:


train_x.shape


# In[89]:


test_x.shape


# In[90]:


train_y.shape


# In[91]:


test_y.shape


# ## Building the model : A comparison between KNN, SVC and Logistic Regression
# 
# 1. K Nearest Neighbors

# In[92]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_x,train_y)


# In[93]:


from sklearn.metrics import accuracy_score

y_pred = knn.predict(test_x)
accuracy_score(y_pred,test_y)


# In[95]:


x_new = [[3.0,266,2,51,2,0,4,2,5.1,3,2,0,4]]

y_new_pred = knn.predict(x_new)
print(y_new_pred)


# 2. Support Vector Classifier

# In[96]:


from sklearn.svm import SVC

svc = SVC(gamma='auto')
svc.fit(train_x,train_y)


# In[97]:


y_p2 = svc.predict(test_x)
accuracy_score(y_p2,test_y)


# 3. Logistic Regression

# In[98]:


from sklearn.linear_model import LogisticRegression

lgr = LogisticRegression(random_state=5,n_jobs=10)
lgr.fit(train_x,train_y)


# In[102]:


y_p3 = lgr.predict(test_x)
accuracy_score(y_p3,test_y)


# In[100]:


y_new_pred2 = lgr.predict(x_new)
print(y_new_pred2)


# [3] corresponds to 'Match'. It is the encoded label. For the given input, x_new, the expected outcome is 'Match'. So, our model is working.

# In[ ]:




