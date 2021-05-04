from data import ImplicitData
import pandas as pd
import numpy as np
from recommenders_implicit import UBISGD,BISGD,ISGD,LocalUBISGD
from eval_implicit import EvalLeaveLastOut
from datetime import datetime

data = pd.read_csv("datasets/ml1m_gte5.csv", "\t")
data.sort_values(by=['Timestamp'], inplace=True, ascending=False)
test_users = []
test_items = []

for user in pd.unique(data['UserID']):
    test_users.append(user)
    item = int(data[data['UserID'] == user][:1]['ItemID'])
    test_items.append(item)
    data.drop(data[(data['UserID'] == user) & (data['ItemID'] == item)].index, inplace=True)

data.sort_values(by=['Timestamp'], inplace=True)

stream = ImplicitData(data['UserID'], data['ItemID'])


print("ml1m 8")

model = ISGD(ImplicitData([],[]),20000, 6, learn_rate = 0.35, u_regularization = 0, i_regularization = 0)

eval = EvalLeaveLastOut(model, stream, test_users, test_items, metrics=["Recall@20"])

start_recommend = datetime.now()
print('start time', start_recommend)

results = eval.EvaluateTime()

print('np.mean(resuls[Recall@20])', np.mean(results['Recall@20']))
print('Length: ', len(results['Recall@20']) )
print('')
print('recommend', np.mean(results['time_recommend']))
print('eval_point', np.mean(results['time_eval_point']))
