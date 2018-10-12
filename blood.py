
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np

train = pd.read_csv("C:\\Users\\Ayush\\Desktop\\Data\\blood\\train.csv")

test = pd.read_csv("C:\\Users\\Ayush\\Desktop\\Data\\blood\\test.csv")


# In[3]:


train.head()


# In[4]:


def outlier25(a):
    b = np.array(a)
    b = np.percentile(b, 25)
    return b
def outlier75(a):
    b = np.array(a)
    b = np.percentile(b, 75)
    return b
outlier_index0 = []
outlier_index1 = []
outlier_index2 = []
outlier_index3 = []
outlier_index4 = []
outlier_index5 = []

for i in range(0,len(train['Unnamed: 0'])):
    IQR = outlier75(train['Unnamed: 0']) - outlier25(train['Unnamed: 0'])
    if train['Unnamed: 0'][i] > outlier75(train['Unnamed: 0'])+1.5*IQR or train['Unnamed: 0'][i] < outlier25(train['Unnamed: 0'])-1.5*IQR:
        outlier_index0.append(i)
print(outlier_index0)


for i in range(0,len(train['Months since Last Donation'])):
    IQR = outlier75(train['Months since Last Donation']) - outlier25(train['Months since Last Donation'])
    if train['Months since Last Donation'][i] > outlier75(train['Months since Last Donation'])+1.5*IQR or train['Number of Donations'][i] < outlier25(train['Number of Donations'])-1.5*IQR:
        outlier_index1.append(i)
print(outlier_index1)

for i in range(0,len(train['Number of Donations'])):
    IQR = outlier75(train['Number of Donations']) - outlier25(train['Number of Donations'])
    if train['Number of Donations'][i] > outlier75(train['Number of Donations'])+1.5*IQR or train['Months since First Donation'][i] < outlier25(train['Months since First Donation'])-1.5*IQR:
        outlier_index2.append(i)
print(outlier_index2)

for i in range(0,len(train['Total Volume Donated (c.c.)'])):
    IQR = outlier75(train['Total Volume Donated (c.c.)']) - outlier25(train['Total Volume Donated (c.c.)'])
    if train['Total Volume Donated (c.c.)'][i] > outlier75(train['Total Volume Donated (c.c.)'])+1.5*IQR or train['Total Volume Donated (c.c.)'][i] < outlier25(train['Total Volume Donated (c.c.)'])-1.5*IQR:
        outlier_index3.append(i)
print(outlier_index3)

for i in range(0,len(train['Months since First Donation'])):
    IQR = outlier75(train['Months since First Donation']) - outlier25(train['Months since First Donation'])
    if train['Months since First Donation'][i] > outlier75(train['Months since First Donation'])+1.5*IQR or train['Months since First Donation'][i] < outlier25(train['Months since First Donation'])-1.5*IQR:
        outlier_index4.append(i)
print(outlier_index4)



# In[5]:


for i in outlier_index1:
    train['Months since Last Donation'][i] = train['Months since Last Donation'].values.mean()


# In[6]:


for i in outlier_index2:
    train['Number of Donations'][i] = train.mode()['Number of Donations'][0]


# In[7]:


for i in outlier_index3:
    train['Total Volume Donated (c.c.)'][i] = train.mode()['Total Volume Donated (c.c.)'][0]


# In[8]:


for i in outlier_index4:
    train['Months since First Donation'][i] = train.mode()['Months since First Donation'][0]


# In[9]:


train.head()


# In[10]:


feature = pd.DataFrame(train['Made Donation in March 2007'])
train = train.drop('Made Donation in March 2007', axis =1 )


# In[11]:


train.dtypes


# In[12]:


train['Unnamed: 0'] = train['Unnamed: 0'].astype(float)
train['Months since Last Donation'] = train['Months since Last Donation'].astype(float)
train['Number of Donations'] = train['Number of Donations'].astype(float)
train['Total Volume Donated (c.c.)'] = train['Total Volume Donated (c.c.)'].astype(float)
train['Months since First Donation'] = train['Months since First Donation'].astype(float)


# In[13]:


train.dtypes


# In[14]:


test['Unnamed: 0'] = test['Unnamed: 0'].astype(float)
test['Months since Last Donation'] = test['Months since Last Donation'].astype(float)
test['Number of Donations'] = test['Number of Donations'].astype(float)
test['Total Volume Donated (c.c.)'] = test['Total Volume Donated (c.c.)'].astype(float)
test['Months since First Donation'] = test['Months since First Donation'].astype(float)


# In[15]:


test.dtypes


# In[16]:


feature['Made Donation in March 2007'] = feature['Made Donation in March 2007'].astype(float)


# In[17]:


feature.dtypes


# In[18]:


from sklearn import preprocessing
x = train.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
train = pd.DataFrame(x_scaled)
train.head()


# In[19]:


for i in range(0,5):
    std = train[i].values.std()
    mean = train[i].values.mean()
    for j in range(0,len(train[i])):
        train[i][j] = (train[i][j] - mean) / std
train.head()


# In[20]:


from sklearn import preprocessing
y = test.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
y_scaled = min_max_scaler.fit_transform(y)
test = pd.DataFrame(y_scaled)
test.head()


# In[21]:


for i in range(0,5):
    std = test[i].values.std()
    mean = test[i].values.mean()
    for j in range(0,len(test[i])):
        test[i][j] = (test[i][j] - mean) / std
test.head()


# In[22]:


from sklearn.svm import LinearSVR
from sklearn.datasets import make_regression
regr = LinearSVR(random_state=0, tol=1e-5)
regr.fit(train, )


# In[23]:


a = clf.predict(test)


# In[24]:


test1 = pd.read_csv("C:\\Users\\Ayush\\Desktop\\Data\\blood\\test.csv")


# In[25]:


result = pd.DataFrame(test1['Unnamed: 0'])


# In[26]:


result['Made Donation in March 2007'] = a


# In[27]:


result.head()


# In[28]:


result['Made Donation in March 2007'] = result['Made Donation in March 2007'].abs()


# In[29]:


result.head()


# In[30]:


result.to_csv("C:\\Users\\Ayush\\Desktop\\Data\\blood\\result9.csv", index = None)

