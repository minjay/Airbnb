import os
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.cross_validation import train_test_split

my_dir = os.getcwd()+'/Airbnb/Data/'
df_train = pd.read_csv(my_dir+'train_users_2.csv')
df_test = pd.read_csv(my_dir+'test_users.csv')
session = pd.read_csv(my_dir+'sessions.csv')

labels = df_train['country_destination'].values
id_test = df_test['id']

df_train.drop(['country_destination'], axis=1, inplace=True)

n_train = df_train.shape[0]
# ignore the old index
df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)

# drop date_first_booking
df_all.drop('date_first_booking', axis=1, inplace=True)

# preprocess age
df_all.loc[df_all['age']>1000, 'age'] = 2015-df_all.loc[df_all['age']>1000, 'age']
df_all.loc[(df_all['age']>100) | (df_all['age']<18), 'age'] = -1
df_all['age'].fillna(-1, inplace=True)

# preprocess date_account_created
# Question: treat dates as continuous or categorical?
dac = np.vstack(df_all['date_account_created'].apply(lambda x: list(map(int, x.split('-')))).values)
df_all['dac_year'] = dac[:, 0]
df_all['dac_month'] = dac[:, 1]
df_all['dac_day'] = dac[:, 2]
df_all.drop('date_account_created', axis=1, inplace=True)

# preprocess timestamp_first_active
tfa = df_all['timestamp_first_active']
df_all['tfa_year'] = tfa.apply(lambda x: int(str(x)[0:4]))
df_all['tfa_month'] = tfa.apply(lambda x: int(str(x)[4:6]))
df_all['tfa_day'] = tfa.apply(lambda x: int(str(x)[6:8]))
# keep hr, min, sec in the model
df_all['tfa_hr'] = tfa.apply(lambda x: int(str(x)[8:10]))
df_all['tfa_min'] = tfa.apply(lambda x: int(str(x)[10:12]))
df_all['tfa_sec'] = tfa.apply(lambda x: int(str(x)[12:14]))
df_all.drop('timestamp_first_active', axis=1, inplace=True)

# OHE
features = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider',
'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
for f in features:
  df_all_dum = pd.get_dummies(df_all[f], prefix=f)
  df_all.drop(f, axis=1, inplace=True)
  df_all = pd.concat([df_all, df_all_dum], axis=1)

# session
session.rename(columns = {'user_id': 'id'}, inplace=True)
# replace NaN by -1
session['action'].fillna(-1, inplace=True)
session['action_type'].fillna(-1, inplace=True)
session['action_detail'].fillna(-1, inplace=True)
action = pd.pivot_table(session, values='secs_elapsed', index='id', columns='action', aggfunc=len,
	fill_value=0)
action.rename(columns=lambda x: 'action_'+str(x), inplace=True)
# action_type does not help
# there are only a few different action_type
# difficult to distinguish
action_detail = pd.pivot_table(session, values='secs_elapsed', index='id', columns='action_detail', aggfunc=len,
	fill_value=0)
action_detail.rename(columns=lambda x: 'action_detail_'+str(x), inplace=True)

# number of different actions for each user
grouped = session[['id', 'action']].groupby('id')
myfun = lambda x: len(pd.Series(x).value_counts())
action_num = grouped.aggregate(myfun)

# combine
df_all = df_all.join(action, on='id')
df_all = df_all.join(action_detail, on='id')
df_all = df_all.join(action_num, on='id')
df_all.fillna(-1, inplace=True)

# drop id
df_all_no_id = df_all.drop('id', axis=1, inplace=False)

X_all = df_all_no_id.values
X = X_all[:n_train, :]
le = LabelEncoder()
y = le.fit_transform(labels)
num_class = max(y)+1

# train test split
# set random seed
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=0)

# fit by xgb
xg_train = xgb.DMatrix(X_train, y_train)
xg_val = xgb.DMatrix(X_val, y_val)

# ndcg5
def ndcg5(preds, dtrain):
  k = 5
  y_true = dtrain.get_label()
  n = len(y_true)
  num_class = preds.shape[1]
  index = np.argsort(preds, axis=1)
  top = index[:, -k:][:,::-1]
  rel = (np.reshape(y_true, (n, 1))==top).astype(int)
  cal_dcg = lambda y: sum((2**y - 1)/np.log2(range(2, k+2)))
  ndcg = np.mean((np.apply_along_axis(cal_dcg, 1, rel)))
  return 'ndcg5', -ndcg

## specify parameters for xgb
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softprob'
param['num_class'] = num_class
param['nthread'] = 20
param['silent'] = 1


evallist  = [(xg_train,'train'), (xg_val,'eval')]
num_round = 1000

# predict
X_test = X_all[n_train:, :]
xg_test = xgb.DMatrix(X_test)
y_pred = bst.predict(xg_test, ntree_limit=bst.best_iteration)
k = 5
index = np.argsort(y_pred, axis=1)
top = le.inverse_transform(index[:, -k:][:,::-1])

# sub
ids = []  #list of ids
cts = np.ravel(top)  #list of countries
for i in range(len(id_test)):
	idx = id_test[i]
	ids += [idx] * 5

sub = pd.DataFrame(data={'id':ids, 'country':cts}, columns=['id', 'country'])
my_dir = os.getcwd()+'/Airbnb/Subs/'
sub.to_csv(my_dir+'sub.csv', index=False)