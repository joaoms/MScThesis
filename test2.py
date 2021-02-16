from data import ImplicitData
import pandas as pd
import numpy as np
from recommenders_implicit import UBISGD,BISGD,ISGD
from eval_implicit import EvalPrequential
from datetime import datetime

data = pd.read_csv("datasets/palco_2010.tsv", "\t")
stream = ImplicitData(data['user_id'], data['track_id'])

print("ml1m 8")

numeroNodes = 8
model = UBISGD(ImplicitData([], []), 200, 6, numeroNodes, learn_rate=0.35,
                                 u_regularization=0.5, i_regularization=0.5)
#model = ISGD(ImplicitData([],[]),200, 6, learn_rate = 0.35, u_regularization = 0.5, i_regularization = 0.5)

eval = EvalPrequential(model, stream, metrics=["Recall@20"])

start_recommend = datetime.now()
print('start time', start_recommend)

results = eval.EvaluateTime(0, stream.size, 1000)

print('np.mean(resuls[Recall@20])', np.mean(results['Recall@20']))
print('Length: ', len(results['Recall@20']) )
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
print('get tuple', np.mean(results['time_get_tuple']))
print('recommend', np.mean(results['time_recommend']))
print('eval_point', np.mean(results['time_eval_point']))
print('update', np.mean(results['time_update']))
