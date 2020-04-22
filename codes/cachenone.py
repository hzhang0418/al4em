'''
cachenone.py
'''
import heapq

import numpy as np
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier

import helper

class CacheNone:
    
    def __init__(self):
        # pairs assigned to this node
        self.pairs = None   # list of (ltable_id, rtable_id)
        self.features = None    # numpy array of features
        
        
    def prepare(self, table_A, table_B, feature_info, pairs):
        self.pairs = pairs
        self.features = np.zeros( (len(self.pairs), len(feature_info)), dtype=np.float32 )
        
        
    def compute_features(self, required_features, feature_info, table_A, table_B):
        if len(required_features)==0:
            return None
        
        # no cache, therefore fetch each pair, then compute required features
        for k, pair in enumerate(self.pairs):
            ltuple = table_A.loc[pair[0]]
            rtuple = table_B.loc[pair[1]]
        
            for f in required_features:
                fs = feature_info.iloc[f]
                lattr = getattr(fs, 'left_attribute')
                rattr = getattr(fs, 'right_attribute')
                ltok = getattr(fs, 'left_attr_tokenizer')
                rtok = getattr(fs, 'right_attr_tokenizer')
                simfunc = nodes.helper.sim_name2func[ getattr(fs, 'simfunction') ]
                
                if ltok==None:
                    value = simfunc(ltuple[lattr], rtuple[rattr])
                else:
                    ltokfunc = nodes.helper.tok_name2func[ltok]
                    rtokfunc = nodes.helper.tok_name2func[rtok]
                    value = simfunc( ltokfunc(ltuple[lattr]), rtokfunc(rtuple[rattr]) )
                
                if np.isnan(value):
                    value = 0
                self.features[k,f] = value
              

    def apply(self, rf: RandomForestClassifier, k: int, exclude_pairs: set) -> list:
        # prediction
        proba = rf.predict_proba(self.features)
        entropies = np.transpose(entropy(np.transpose(proba), base=2))
        
        # select top k, return list of pairs of (index, entropy)
        candidates = [ (self.pairs[k],v)  for k,v in enumerate(entropies) if self.pairs[k] not in exclude_pairs ]
        top_k = heapq.nlargest(k, candidates, key=lambda p: p[1])
        
        return top_k       
    
    