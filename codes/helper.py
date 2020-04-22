'''
helper.py
'''
import random 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import py_entitymatching as em

tok_name2func = em.get_tokenizers_for_matching(q = [2,3,4,5])
sim_name2func = em.get_sim_funs_for_matching()

def get_features_in_random_forest(rf: RandomForestClassifier) -> list:
    rf_features = set()
    for tree in rf.estimators_:
        for f in tree.tree_.feature:
            if f != -2:
                rf_features.add(f)
    return list(rf_features)

def get_features_in_decision_tree(tree: DecisionTreeClassifier) -> list:
    tree_features = set()
    for f in tree.tree_.feature:
        if f != -2:
            tree_features.add(f)
    return list(tree_features)
