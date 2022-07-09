#!/usr/bin/env python
# coding: utf-8

# # Arjun ML-Assignment-2

# In[1]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.compose import make_column_selector as selector
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier


# # Reading Dataset

# In[2]:


colnames= ['Age','workclass','fnlwgt','education','education-num','marital-status','occupation',
           'relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','salary']
data = pd.read_csv('adult.data', sep=",", header=None, names=colnames)



# In[3]:


data.isnull().sum()


# # Pre-Processing

# In[4]:


data.salary.unique()


# setting values as 0 and 1 in the prediction column as it has only two values mentioned above

# In[5]:


data["salary"] = data["salary"].apply(lambda x: 0 if x == ' <=50K'  else 1)
data.head()


# In[6]:


data.salary.unique()


# #label encoding can also be done
# #from sklearn.preprocessing import LabelEncoder
# #le = LabelEncoder()
# #data['salary'] = le.fit_transform(data['salary'])
# #data.head()

# In[7]:


#rows with salary <=50K and salary >50K
data['salary'].value_counts()


# Similarly, label encoding of column name 'sex'

# In[8]:


le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])



# In[9]:


workclass_data = data['workclass'].value_counts()



# In[10]:


import matplotlib.pyplot as plt

#plt.figure(figsize = (15, 6))
 
# creating the bar plot
#plt.bar(workclass_data.index,workclass_data, color ='maroon',width = 0.4)
#plt.xlabel("workclass", fontsize=15)
#plt.ylabel("Count")
#plt.title("Count of people in each workclass")


# Replacing '?' with Nan values

# In[11]:


data.loc[data["workclass"] == " ?", "workclass"] = np.nan
data['workclass'].value_counts()


# In[12]:


education_data = data['education'].value_counts()



# In[13]:


#plt.figure(figsize = (15, 6))
# creating the bar plot
#plt.bar(education_data.index, education_data, color ='maroon',width = 0.4)
#plt.xlabel("Education", fontsize=15)
#plt.ylabel("Count")
#plt.title("Count of people with differnt degrees")


# In[14]:


marital_data = data['marital-status'].value_counts()



# In[15]:


#plt.figure(figsize = (15, 6))
# creating the bar plot
#plt.bar(marital_data.index, marital_data, color ='maroon',width = 0.4)
#plt.xlabel("Marital-status", fontsize=15)
#plt.ylabel("Count")
#plt.title("Count of people with different marital-status")


# In[16]:


occupation_data = data['occupation'].value_counts()



# In[17]:


#plt.figure(figsize = (25, 6))
# creating the bar plot
#plt.bar(occupation_data.index, occupation_data, color ='maroon',width = 0.4)
#plt.xlabel("Occupation", fontsize=15)
#plt.ylabel("Count")
#plt.title("Count of people with different Occupation")


# Replacing '?' with Nan

# In[18]:


data.loc[data["occupation"] == " ?", "occupation"] = np.nan
data['occupation'].value_counts()


# In[19]:


relationship_data = data['relationship'].value_counts()



# In[20]:


#plt.figure(figsize = (15, 6))
# creating the bar plot
#plt.bar(relationship_data.index, relationship_data, color ='maroon',width = 0.4)
#plt.xlabel("Relationship Status", fontsize=15)
#plt.ylabel("Count")
#plt.title("Count of people with different Relationship Status")


# In[21]:


race_data = data['race'].value_counts()


# In[22]:


#plt.figure(figsize = (15, 6))
# creating the bar plot
#plt.bar(race_data.index, race_data, color ='maroon',width = 0.4)
#plt.xlabel("Race", fontsize=15)
#plt.ylabel("Count")
#plt.title("Count of people with different Race")


# In[23]:


nc_data = data['native-country'].value_counts()



# In[24]:


#plt.figure(figsize = (35, 10))
# creating the bar plot
#plt.bar(nc_data.index, nc_data, color ='maroon',width = 0.4)
#plt.xlabel("Native-Country", fontsize=15)
#plt.ylabel("Count")
#plt.title("Count of people from different countries")


# As counts of United states is way more than other countries so label encoding would not make sense hence dividing the column in two values 'US' and 'Others'

# In[25]:


data["native-country"] = data["native-country"].apply(lambda x: 'US' if x == ' United-States'  else 'Others')
data["native-country"].value_counts()


# In[26]:


data.isnull().sum()


# As null values count is 1836 which is 5.06% of 32561 total rows we can drop these rows

# In[27]:


data.dropna(inplace=True)


# In[28]:





# In[29]:


data['education'].unique()


# In[30]:


le = LabelEncoder()
data['education'] = le.fit_transform(data['education'])
data['education'].unique()


# Now, Label encoding of all the remaining categorical columns

# In[31]:


categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)



# In[32]:


for i in categorical_columns:
    le = LabelEncoder()
    data[i] = le.fit_transform(data[i])


# In[33]:


#data.head()


# In[34]:


x=data.drop(['salary'], axis=1)
y=data.iloc[:,-1]



# In[35]:





# Splitting the dataset

# In[36]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


# Feature Scaling  

# In[37]:


from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)  


# # Model Building

# In[38]:


lgbm = LGBMClassifier()
lgbm.fit(x_train, y_train)


# In[39]:


pred = lgbm.predict(x_test)


# In[40]:


df = pd.DataFrame(list(zip(y.values, pred)), columns =['Actual', 'Predicted'])



# In[41]:


lgbm_acc = accuracy_score(y_test, pred)*100
print("Accuracy Score: ", lgbm_acc)


# In[42]:


from sklearn.linear_model import LogisticRegression  
classifier= LogisticRegression(random_state=0)  
classifier.fit(x_train, y_train)


# In[43]:


lr_pred= classifier.predict(x_test)
lr_df = pd.DataFrame(list(zip(y.values, lr_pred)), columns =['Actual', 'Predicted'])



# In[44]:


lr_acc = accuracy_score(y_test, lr_pred)*100
print("Accuracy Score: ", lr_acc)


# In[45]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)


# In[46]:


nb_pred= nb.predict(x_test)
nb_df = pd.DataFrame(list(zip(y.values, nb_pred)), columns =['Actual', 'Predicted'])



# In[47]:


nb_acc = accuracy_score(y_test, nb_pred)*100
print("Accuracy Score: ", nb_acc)


# In[48]:


from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(y_test, nb_pred)  



# In[49]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(x_train,y_train)


# In[50]:


knn_pred= knn.predict(x_test)
knn_df = pd.DataFrame(list(zip(y.values, knn_pred)), columns =['Actual', 'Predicted'])



# In[51]:


knn_acc = accuracy_score(y_test, knn_pred)*100
print("Accuracy Score: ", knn_acc)


# In[52]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(max_depth=10, random_state = 42, max_features = None, min_samples_leaf = 15)
dtree.fit(x_train,y_train)


# In[53]:


dtree_pred= dtree.predict(x_test)
dtree_acc = accuracy_score(y_test, dtree_pred)*100
print("Accuracy Score: ", dtree_acc)


# In[54]:


from sklearn.ensemble import RandomForestClassifier
rfm = RandomForestClassifier(n_estimators = 70, n_jobs=-1, random_state = 42, min_samples_leaf=30)
rfm.fit(x_train,y_train)


# In[55]:


rfm_pred= rfm.predict(x_test)
rfm_acc = accuracy_score(y_test, rfm_pred)*100
print("Accuracy Score: ", rfm_acc)


# In[56]:


from sklearn.svm import SVC
svc = SVC(kernel = 'linear', C=0.025, random_state=42)
svc.fit(x_train,y_train)


# In[57]:


svc_pred= svc.predict(x_test)
svc_acc = accuracy_score(y_test, svc_pred)*100
print("Accuracy Score: ", svc_acc)


# Creating dataframe with all the models accuracy

# In[58]:


acc_df = pd.DataFrame({'Accuracy':[lr_acc, nb_acc, knn_acc, dtree_acc, rfm_acc, svc_acc, lgbm_acc]},
                         index=['Logistic Regression','Gaussian Naive Bayes','KNn Classifier','Decision tree',
                                'Random Forest', 'Support Vector Classifier', 'LightGBM'])
acc_df.index.name = 'Classification Models'



# In[84]:


f = plt.figure()
f.set_figwidth(15)
f.set_figheight(7)
plt.bar(acc_df.index,acc_df['Accuracy'])
plt.title('Accuracy of Models')
plt.xlabel('Classification Models')
plt.ylabel('Accuracy Percentage')
plt.savefig(r'static\classification.png')
#plt.show()


# From the above barplot we can see LGBM has higher accuracy among all so we can use this model for classification

# # Applying trained models on test data given in question

# In[60]:


colnames= ['Age','workclass','fnlwgt','education','education-num','marital-status','occupation',
           'relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','salary']
test_data =  pd.read_csv(r'D:\Term3\Machine Learning\adult.test', sep=",", header=None, names=colnames)
test_data.head()


# In[61]:


#dropping first row
test_data = test_data.iloc[1: , :]
test_data.head()


# In[62]:



# # Same Pre Processing steps for test data

# In[63]:


test_data["salary"] = test_data["salary"].apply(lambda x: 0 if x == ' <=50K.'  else 1)
test_data.head()


# In[64]:


test_data.isin([' ?']).any()


# In[65]:


test_data.loc[test_data["workclass"] == " ?", "workclass"] = np.nan


# In[66]:


test_data.loc[test_data["occupation"] == " ?", "occupation"] = np.nan


# In[67]:


le = LabelEncoder()
test_data['sex'] = le.fit_transform(test_data['sex'])
test_data.head()


# In[68]:


nc_data = test_data['native-country'].value_counts()


# In[69]:


test_data["native-country"] = test_data["native-country"].apply(lambda x: 'US' if x == ' United-States'  else 'Others')
test_data["native-country"].value_counts()


# In[70]:


test_data.isnull().sum()


# In[71]:


test_data.dropna(inplace=True)


# In[72]:


categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(test_data)



# Converting Age column from string to integer

# In[73]:


test_data['Age'] = test_data['Age'].astype('int')


# In[74]:


categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(test_data)



# In[75]:


for i in categorical_columns:
    le = LabelEncoder()
    test_data[i] = le.fit_transform(test_data[i])


# In[76]:


X=test_data.drop(['salary'], axis=1)
Y=test_data.iloc[:,-1]



# In[77]:





# # Applying LGBM model to predict on test data as it has higher accuracy

# In[78]:


sd= StandardScaler()    
X= sd.fit_transform(X)     


# In[79]:


test_pred = lgbm.predict(X)
test_lgbm_acc = accuracy_score(Y, test_pred)*100
print("Accuracy Score: ", test_lgbm_acc)


# Checking Naive bayes also but LGBM accuracy is still higher

# In[80]:


test_nb_pred = nb.predict(X)
test_nb_acc = accuracy_score(Y, test_nb_pred)*100
print("Accuracy Score: ", test_nb_acc)


# In[81]:


final_df = pd.DataFrame(list(zip(Y.values, test_pred)), columns =['Actual', 'Predicted'])



# In[82]:


test_cm= confusion_matrix(Y, test_pred)  



# Therefore 10742+2402 = 13144 rows are corectly predicted

# In[83]:


import numpy as np
unique, counts = np.unique(test_pred, return_counts=True)
a = dict(zip(unique, counts))



# from the above dictionary we can say that our model predicted 12112 people having salary <=50k and 3203 people having salary >50K

# In[ ]:




