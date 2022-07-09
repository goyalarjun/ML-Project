#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, PolynomialFeatures
from sklearn.compose import make_column_selector as selector
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint

# In[2]:


data = pd.read_csv('housing.csv')


# In[3]:


data.isnull().sum()

# In[4]:


data['total_bedrooms'].mean()

# In[5]:


data.fillna(537.8705525375618, inplace=True)

# In[6]:


data.isnull().sum()

# In[7]:


x = data.drop(['median_house_value'], axis=1)
y = data.iloc[:, -2].values


# In[8]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# In[9]:




# In[10]:


categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)


# In[11]:


categorical_preprocessor = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
preprocessor = ColumnTransformer([
    ('cat_preprocessor', categorical_preprocessor, categorical_columns)],
    remainder='passthrough', sparse_threshold=0)

# In[12]:


model_pipeline = Pipeline([("preprocessor", preprocessor), ("sc", StandardScaler()),
                           ("random_forest", RandomForestRegressor(random_state=42))])
base_rf_model = model_pipeline.fit(x_train, y_train)
y_pred = base_rf_model.predict(x_test)

# In[13]:


new_df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})


# In[14]:


# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(RandomForestRegressor().get_params().keys())

# In[15]:


import numpy as np



# In[17]:


# pipe =Pipeline([("preprocessor", preprocessor),("sc",StandardScaler(),("regression",'reg'))])
parameters = {
    'random_forest__bootstrap': [True, False],
    'random_forest__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
    'random_forest__max_features': ['auto', 'sqrt'],
    'random_forest__min_samples_leaf': [1, 2, 4],
    'random_forest__min_samples_split': [2, 5, 10],
    'random_forest__n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
}
rand_search = RandomizedSearchCV(model_pipeline, param_distributions=parameters, n_iter=3, cv=2, verbose=2,
                                 random_state=42)
rand_search.fit(x_train, y_train)


# In[92]:


def evaluate(model, x_test, y_test):
    predictions = model.predict(x_test)
    errors = abs(predictions - y_test)
    mape = 100 * np.mean(errors / y_test)
    accuracy = 100 - mape
    return accuracy


# In[19]:




# In[20]:



# In[21]:


best_random = rand_search.best_estimator_


# In[22]:


svr_model_pipeline = Pipeline([("preprocessor", preprocessor), ("sc", StandardScaler()), ("svr", SVR())])
svr_base_model = svr_model_pipeline.fit(x_train, y_train)
svr_pred = svr_base_model.predict(x_test)


# In[23]:



# neg_mean_absolute_percentage_error


# In[5]:




# In[25]:


# defining parameter range
svr_parameters = {'svr__C': [0.1, 1, 10, 100, 1000],
                  'svr__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'svr__kernel': ['rbf']}  # poly,sigmoid,linear

# In[26]:


svr_rand_search = RandomizedSearchCV(svr_model_pipeline, param_distributions=svr_parameters, n_iter=20, cv=3,
                                     random_state=42, n_jobs=-1)
svr_rand_search.fit(x_train, y_train)

# In[27]:


svr_best_random = svr_rand_search.best_estimator_


# In[28]:


dtr_model_pipeline = Pipeline(
    [("preprocessor", preprocessor), ("sc", StandardScaler()), ("dtr", DecisionTreeRegressor())])
dtr_base_model = dtr_model_pipeline.fit(x_train, y_train)
dtr_pred = dtr_base_model.predict(x_test)


# In[29]:

dtr_parameters = {"dtr__splitter": ["best", "random"],
                  "dtr__max_depth": [None, 1, 3, 5, 7, 9, 11, 12],
                  "dtr__min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  "dtr__min_weight_fraction_leaf": [0.001, 0.005, 0.00089, 0.05, 0.00081, 0.2, 0.3],
                  "dtr__max_features": ["auto", "log2", "sqrt", None],
                  "dtr__max_leaf_nodes": [None, 10, 20, 30, 40, 50, 60, 70, 80, 90]}

# In[72]:


dtr_rand_search = RandomizedSearchCV(dtr_model_pipeline, param_distributions=dtr_parameters, n_iter=60, cv=3,
                                     random_state=42, n_jobs=-1, verbose=3)
dtr_rand_search.fit(x_train, y_train)

# In[73]:




# In[74]:


dtr_best_random = dtr_rand_search.best_estimator_
evaluate(dtr_best_random, x_test, y_test)

# In[35]:


lr_model_pipeline = Pipeline([("preprocessor", preprocessor), ("sc", StandardScaler()), ("lr", LinearRegression())])
lr_base_model = lr_model_pipeline.fit(x_train, y_train)
lr_pred = lr_base_model.predict(x_test)

# In[36]:



# In[37]:


lasso_model_pipeline = Pipeline([("preprocessor", preprocessor), ("sc", StandardScaler()), ("lasso", Lasso())])
lasso_base_model = lasso_model_pipeline.fit(x_train, y_train)
lasso_pred = lasso_base_model.predict(x_test)

# In[38]:




# In[39]:


pprint(Lasso().get_params())

# In[40]:


lasso_params = {'lasso__alpha': [0.02, 0.024, 0.025, 0.026, 0.03, 0.11, 0.08, 0.09]}
ridge_params = {'ridge__alpha': [200, 20, 35, 250, 150, 270, 110, 45, 60, 500, 190, 210, 170]}

# In[41]:


lasso_grid_search = GridSearchCV(lasso_model_pipeline, param_grid=lasso_params, n_jobs=-1, cv=3,
                                 scoring='neg_mean_absolute_percentage_error')
lasso_grid_search.fit(x_train, y_train)

# In[42]:




# In[43]:


lasso_best_grid = lasso_grid_search.best_estimator_
evaluate(lasso_best_grid, x_test, y_test)

# In[44]:


ridge_model_pipeline = Pipeline([("preprocessor", preprocessor), ("sc", StandardScaler()), ("ridge", Ridge())])
ridge_base_model = ridge_model_pipeline.fit(x_train, y_train)
ridge_pred = ridge_base_model.predict(x_test)


# In[45]:




# In[46]:


ridge_grid_search = GridSearchCV(ridge_model_pipeline, param_grid=ridge_params, n_jobs=-1, cv=3, scoring='r2')
ridge_grid_search.fit(x_train, y_train)

# In[47]:


# In[48]:


ridge_best_grid = ridge_grid_search.best_estimator_
evaluate(ridge_best_grid, x_test, y_test)

# In[49]:


en_model_pipeline = Pipeline([("preprocessor", preprocessor), ("sc", StandardScaler()), ("EN", ElasticNet())])
en_base_model = en_model_pipeline.fit(x_train, y_train)
en_pred = en_base_model.predict(x_test)


# In[50]:




# In[51]:


en_params = {"EN__max_iter": [1, 10, 20, 40],
             "EN__alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
             "EN__l1_ratio": np.arange(0.0, 1.0, 0.1)}

# In[52]:


en_grid_search = GridSearchCV(en_model_pipeline, param_grid=en_params, n_jobs=-1, cv=10,
                              scoring='neg_mean_absolute_percentage_error')
en_grid_search.fit(x_train, y_train)

# In[53]:



# In[54]:


en_best_grid = en_grid_search.best_estimator_
evaluate(en_best_grid, x_test, y_test)

# In[55]:


pipe1 = Pipeline([("preprocessor", preprocessor), ("sc", StandardScaler()), ('poly', PolynomialFeatures()),
                  ('fit', LinearRegression())])
pipe2 = Pipeline(
    [("preprocessor", preprocessor), ("sc", StandardScaler()), ('poly', PolynomialFeatures()), ('fit', Lasso())])
pipe3 = Pipeline(
    [("preprocessor", preprocessor), ("sc", StandardScaler()), ('poly', PolynomialFeatures()), ('fit', Ridge())])

# In[56]:


lasso_poly = pipe2.fit(x_train, y_train)
lasso_poly_pred = lasso_poly.predict(x_test)


# In[57]:




# In[58]:


ridge_poly = pipe3.fit(x_train, y_train)
ridge_poly_pred = ridge_poly.predict(x_test)


# In[59]:




# In[60]:


ols_poly = pipe1.fit(x_train, y_train)
ols_poly_pred = ols_poly.predict(x_test)


# In[61]:




# There is high increment in the accuracy for lasso, ridge and linear regressions.
# Lasso and Ridge performs better with high dimmensional data.

# In[101]:


scores_df = pd.DataFrame({'Base model': [evaluate(svr_base_model, x_test, y_test),
                                         evaluate(base_rf_model, x_test, y_test),
                                         evaluate(dtr_base_model, x_test, y_test),
                                         evaluate(lr_base_model, x_test, y_test),
                                         evaluate(lasso_base_model, x_test, y_test),
                                         evaluate(ridge_best_grid, x_test, y_test),
                                         evaluate(en_base_model, x_test, y_test), evaluate(ols_poly, x_test, y_test),
                                         evaluate(lasso_poly, x_test, y_test), evaluate(ridge_poly, x_test, y_test)],
                          'Tuned Model': [evaluate(svr_best_random, x_test, y_test),
                                          evaluate(best_random, x_test, y_test),
                                          evaluate(dtr_best_random, x_test, y_test), 0,
                                          evaluate(lasso_best_grid, x_test, y_test),
                                          evaluate(ridge_best_grid, x_test, y_test),
                                          evaluate(en_best_grid, x_test, y_test), 0, 0, 0]},
                         index=['SVR', 'RFR', 'DTR', 'Linear Regression', 'Lasso', 'Ridge', 'Elastic Net', 'LR Poly',
                                'Lasso Poly', 'Ridge Poly'])


# In[95]:


import matplotlib.pyplot as plt

f = plt.figure()
f.set_figwidth(15)
f.set_figheight(7)
plt.bar(scores_df.index, scores_df['Base model'])
plt.title('Accuracy of Models')
plt.xlabel('Regression Models')
plt.ylabel('Accuracy Percentage')

# In[112]:


fig, ax = plt.subplots()
fig.set_figwidth(15)
fig.set_figheight(7)

index = np.arange(10)
bar_width = 0.35
base = ax.bar(index - 0.2, scores_df['Base model'], bar_width, label='Base Model')

tuned = ax.bar(index + 0.2, scores_df['Tuned Model'], bar_width, label='Tuned Model')

ax.set_xlabel('Regression Models')
ax.set_ylabel('Accuracy Percentage')
ax.set_title('Accuracy of Models')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(scores_df.index)
ax.legend()
plt.savefig(r'static\regression.png')

