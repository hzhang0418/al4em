'''
driver.py
'''
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

import py_entitymatching as em

import helper

class Driver:
    
    def __init__(self):
        # table A, B and candidate pairs
        self.table_A = None
        self.table_B = None
        
        # feature table
        self.feature_table = None
        self.tok_name2func = None
        self.sim_name2func = None
        
        # features and labels, used for training model
        self.features = None
        self.labels = None
        
        # labeled pairs
        self.pair2label = {}
        
    def prepare(self, table_A, table_B, feature_table, tok_name2func, sim_name2func):
        self.table_A = table_A
        self.table_B = table_B
        self.feature_table = feature_table
        self.tok_name2func = tok_name2func
        self.sim_name2func = sim_name2func
                
    def add_new_training(self, pair2label: dict) -> None:
        pairs = []
        labels = []
        for k,v in pair2label.items():
            if k not in self.pair2label: # only need to compute features for new pairs
                pairs.append(k)
                labels.append(v)
            self.pair2label[k] = v
            
        self._compute_features(pairs)
        if self.labels is None:
            self.labels = np.array(labels, dtype=int)
        else:
            self.labels = np.hstack( (self.labels, np.array(labels, dtype=int)) )
        
    def _compute_features(self, pairs: list) -> None:
        
        features_new = np.empty( (len(pairs), len(self.feature_table)), dtype=np.float32)
        try:
            f = 0
            for fs in self.feature_table.itertuples(index=False):
                lattr = getattr(fs, 'left_attribute')
                rattr = getattr(fs, 'right_attribute')
                ltok = getattr(fs, 'left_attr_tokenizer')
                rtok = getattr(fs, 'right_attr_tokenizer')
                simfunc = self.sim_name2func[ getattr(fs,'simfunction') ]
                #func = getattr(fs, 'function') 
    
                if ltok is None:
                    for k, pair in enumerate(pairs):
                        ltable_id, rtable_id = pair[0], pair[1]
                        ltable_value = self.table_A.loc[ltable_id][lattr]
                        rtable_value = self.table_B.loc[rtable_id][rattr]
                        features_new[k][f] = simfunc(ltable_value, rtable_value)
                else:
                    ltokfunc = self.tok_name2func[ltok]
                    rtokfunc = self.tok_name2func[rtok]
                    for k, pair in enumerate(pairs):
                        ltable_id, rtable_id = pair[0], pair[1]
                        ltable_value = self.table_A.loc[ltable_id][lattr]
                        rtable_value = self.table_B.loc[rtable_id][rattr]
                        features_new[k][f] = simfunc(ltokfunc(ltable_value), rtokfunc(rtable_value))
                f += 1
        except ValueError:
            print(pair, ltable_value, rtable_value)
            raise
                    
        np.nan_to_num(features_new, copy=False)
        
        if self.features is None:
            self.features = features_new
        else:            
            self.features = np.vstack( (self.features, features_new) )
        
    def train(self) -> RandomForestClassifier:
        #features, labels = self.features, self.labels
        #features, labels = shuffle(self.features, self.labels)
        features, labels = shuffle(self.features, self.labels, random_state=0)
        rf = RandomForestClassifier(n_estimators=10, max_depth=None, max_features='auto', random_state=0, n_jobs=1)
        #rf = DecisionTreeClassifier(random_state=0)
        rf.fit(features, labels)
        return rf                  
                    