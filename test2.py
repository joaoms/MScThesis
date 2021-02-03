from data import ImplicitData
from pprint import pprint
import pandas as pd
from operator import itemgetter
import numpy as np
from recommenders_implicit import BISGD,ISGD
from eval_implicit import EvalPrequential
from datetime import datetime

data = pd.read_csv("datasets/palco_2010.tsv","\t")
stream = ImplicitData(data['user_id'],data['track_id'])

print("ml1m 8")

numeroNodes = 8
model = BISGD(ImplicitData([],[]),200, 6, numeroNodes, learn_rate = 0.35, u_regularization = 0.5, i_regularization = 0.5, use_numba = False)
#model = ISGD(ImplicitData([],[]),160, 8, learn_rate = 0.1, u_regularization = 0.4, i_regularization = 0.4, use_numba = False)

eval = EvalPrequential(model,stream, metrics = ["Recall@20"])

start_recommend = datetime.now()
print('start time', start_recommend)

results=eval.EvaluateTime(stream.size - 1000,stream.size)

print('npmean(resuls[Recall@20])', np.mean(results['Recall@20']))

#1 node
#0.18298
#runtime 2:13:07

#recall20 n=2
# 0.1951
#run time 3:40:22

#6 nodes
#recall20 = 0.2022
#run time = 9:09:32

#8 nodes
#recall20= 0.2040
# run time 12:16:38

#12 nodes
#recall20= 0.20316
# run time 17:58:27

#recall20 ISGD= 0.256

end_recommend = datetime.now()
print('end time', end_recommend)

tempo = end_recommend - start_recommend

print('run time', tempo)
print('')
print('get tuple',np.mean(results['time_get_tuple']))
print('recommend',np.mean(results['time_recommend']))
print('eval_point',np.mean(results['time_eval_point']))
print('update',np.mean(results['time_update']))
