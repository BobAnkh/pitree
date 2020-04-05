import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
import pydotplus
import pickle as pk
import csv
import pensieve
import pensiedt
import robustmpc
import robustmdt
import hotdash
import hotdadt
import argparse
import load_trace
import fixed_env as env
import fixed_env_hotdash as env_hotdash
from multiprocessing import Pool
import time
from sklearn.datasets import load_iris
from sklearn.tree import tree
from sklearn_porter import Porter


with open('./results/decision_tree/robustmpc_lin_mixed_500.pk3','rb') as f:
  best_tree=pk.load(f)
porter=Porter(best_tree,language='js')
output=porter.export(embed_data=True)
with open('./results/decision_tree/dt3','w') as fp:
  fp.write(output)
# print(output)