#!/usr/bin/env python
# coding: utf-8

# In[20]:


#!pip install --upgrade implicit
import numpy as np
import pandas as pd
import pickle
import os; os.environ['OPENBLAS_NUM_THREADS']='1'
from implicit.als import AlternatingLeastSquares as ALS
from implicit.evaluation import mean_average_precision_at_k
from scipy.sparse import coo_matrix


# In[2]:


# Loading Data frames
base_path = '/Users/saipavan/Downloads/Datazoids/'
#base_path = '/Users/bhadrashah/Downloads/'
train_csv = f'{base_path}transactions_train.csv.zip'
users_path = f'{base_path}customers.csv.zip'
articles_path = f'{base_path}articles.csv.zip'
df = pd.read_csv(train_csv, dtype={'article_id': str}, parse_dates=['t_dat'])
users = pd.read_csv(users_path)
articles = pd.read_csv(articles_path, dtype={'article_id': str})


# In[3]:


# autoincrementing ids starting from 0 to both users and items
all_users = users['customer_id'].unique().tolist()
all_items = articles['article_id'].unique().tolist()
user_ids_dict = dict(list(enumerate(all_users)))
item_ids_dict = dict(list(enumerate(all_items)))
users_map = {user_uid: user_idx for user_idx, user_uid in user_ids_dict.items()}
items_map = {item_uid: item_idx for item_idx, item_uid in item_ids_dict.items()}
df['user_id'] = df['customer_id'].map(users_map)
df['item_id'] = df['article_id'].map(items_map)

del users, articles, user_ids_dict, item_ids_dict


# In[4]:


# creating a sparse matrix using coo_matrix function in users x item format.
rows = df['user_id'].values
columns = df['item_id'].values
data = np.ones(df.shape[0])
coo_train_matrix = coo_matrix((data, (rows, columns)), shape=(len(all_users), len(all_items)))
coo_train_matrix


# In[5]:


# basic check for model and data compatability
model = ALS(iterations=2,factors=10)
model.fit(coo_train_matrix)


# In[6]:


# Functions 
def split_dataset(df, validation_days=7):
    # Split a pandas dataframe into training and validation data, based on validation_days
    data_split_time = df['t_dat'].max() - pd.Timedelta(validation_days)
    train_df = df[df['t_dat'] < data_split_time]
    val_df = df[df['t_dat'] >= data_split_time]
    return train_df, val_df

def user_item_coo_matrix(df):
    # Turn a dataframe with transactions into a COO sparse matrix of items x users format
    rows = df['user_id'].values
    columns = df['item_id'].values
    data = np.ones(df.shape[0])
    coo_m = coo_matrix((data, (rows, columns)), shape=(len(all_users), len(all_items)))
    return coo_m


def csr_matrices(df, validation_days=7):
   
    train_df, val_df = split_dataset(df, validation_days=validation_days)
    coo_train_matrix = user_item_coo_matrix(train_df)
    coo_value = user_item_coo_matrix(val_df)

    csr_train_matrix = coo_train_matrix.tocsr()
    csr_value = coo_value.tocsr()
    
    return {'coo_train_matrix': coo_train_matrix, 'csr_train_matrix': csr_train_matrix, 'csr_value': csr_value}


def validation(matrices_temp, iterations=20, factors=200, regularization=0.01):
    csr_train_matrix=matrices_temp['csr_train_matrix']
    csr_value =  matrices_temp['csr_value']
    coo_train_matrix=matrices_temp['coo_train_matrix']
    model = ALS(factors=factors,  regularization=regularization, random_state=42,iterations=iterations)
    model.fit(coo_train_matrix)
    map_12 = mean_average_precision_at_k(model, csr_train_matrix, csr_value, K=12, num_threads=4)
    print(f"Factors: {factors:>3}")
    print(f"Iterations: {iterations:>2}" )
    print(f"Regularization: {regularization:4.3f} ")
    print(f"MAP 12: {map12:6.9f}")
    return map_12



# In[7]:


new_df = df[df['t_dat'] > '2020-08-21']
matrices =  csr_matrices(new_df)


# In[8]:


get_ipython().run_cell_magic('time', '', '# alternating the parameters to get the best model parameters\nbest_map = 0\nfor factors in [40, 50, 60, 100, 200, 500]:\n    for iterations in [3, 12, 15, 20]:\n        map12 = validate(matrices, factors, iterations, 0.01, show_progress=False)\n        if map12 > best_map:\n            best_map = map12\n            best_params = {\'factors\': factors, \'iterations\': iterations, \'regularization\': 0.01}\n            print(f"Best MAP found. The new best parameters are: {best_params}")')


# In[9]:


map12 = validate(matrices, factors=200, iterations=3, regularization=0.01, show_progress=True)
print(map12)


# In[10]:


del matrices


# In[11]:


# Training over the full dataset
coo_train_matrix = user_item_coo_matrix(df)
csr_train_matrix = coo_train_matrix.tocsr()


# In[12]:


best_params = {"factors":200, "iterations":15, "regularization":0.01, "show_progress":True}


# In[14]:


model = ALS(factors=200, regularization=0.01,iterations=15, random_state=50)
model.fit(coo_train_matrix, show_progress=True)
def output(model, csr_train_matrix, file_name):
    predicted_value = []
    batch_size = 2000
    users_indices = np.arange(len(all_users))
    for start_idx in range(0, len(users_indices), batch_size):
        batch = users_indices[start_idx : start_idx + batch_size]
        ids, scores = model.recommend(batch, csr_train_matrix[batch], N=6, filter_already_liked_items=False)
        for i, userid in enumerate(batch):
            customer_id = all_users[userid]
            user_items = ids[i]
            article_ids = [all_items[item_id] for item_id in user_items]
            preddicted_value.append((customer_id, ' '.join(article_ids)))

    data_predicted = pd.DataFrame(predicted_value, columns=['customer_id', 'prediction'])
    data_predicted.to_csv(file_name, index=False)
    
    display(data_predicted.head())
    print(data_predicted.shape)
    
    return data_predicted


# In[15]:


data_pred = submit(model, csr_train, "submissions-Implicit-ALS.csv");


# In[16]:


new_df = df.drop(['t_dat', 'sales_channel_id', 'price', 'user_id', 'item_id'],  axis=1)
new_df


# In[17]:


new_df = new_df.groupby('customer_id')['article_id'].apply(list).reset_index(name="article_id")
new_df


# In[21]:


# This data will be used in app.py (streamlit app) to know the customers list and the items that they have bought.
pickle.dump(new_df, open('customer_transactions_embeddings.pkl','wb'))


# In[23]:


np.random.choice(new_df['customer_id'])


# In[24]:


np.random.choice(new_df['customer_id'])

