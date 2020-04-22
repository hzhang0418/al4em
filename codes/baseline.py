'''
Baseline solution
'''
import os
import copy
import heapq
import numpy as np
import time
import pandas as pd
from pyspark import SparkContext

import seeder
import labeler
import driver
import cachenone
import helper

def run(sc, table_A, table_B, candidate_pairs, table_G, feature_table, feature_info,
		num_executors, seed_size, max_iter, batch_size):
 
    # prepare driver
    # driver node
    driver = driver.Driver()
    driver.prepare(table_A, table_B, feature_table, helper.tok_name2func, 
		helper.sim_name2func)
    
    # seeds
    seeder = seeder.Seeder(table_G)
    labeler = labeler.Labeler(table_G)
    
    # partition pairs
    pair_rdd = sc.parallelize(candidate_pairs, num_executors) 
    bc_table_A = sc.broadcast(table_A)
    bc_table_B = sc.broadcast(table_B)
    bc_feature_info = sc.broadcast(feature_info)
    
	# compute feature vectors
    ex_rdd = pair_rdd.mapPartitions(
        lambda pairs_partition: create_executors(pairs_partition, bc_table_A, 
		bc_table_B, bc_feature_info, num_executor, cache_level), 
        preservesPartitioning=True)
    ex_rdd.cache()
    
    # simulate active learning    
    # select seeds
    pair2label = seeder.select(seed_size)
    exclude_pairs = set(pair2label.keys())
    
    num_iter = 0
    all_features = set()

    while num_iter<max_iter:
        driver.add_new_training(pair2label)
        # train model
        rf = driver.train()
        # features in RF
        required_features = nodes.helper.get_features_in_random_forest(rf)
        all_features.update(required_features)
        
        # select most informative examples
        candidates = ex_rdd.mapPartitions(
            lambda executors: iteration(executors, rf, batch_size, exclude_pairs), 
			preservesPartitioning=True).collect()
        
        # select top k from candidate
        top_k = heapq.nlargest(batch_size, candidates, key=lambda p: p[1])
        top_k_pairs = [ t[0] for t in top_k ]
        pair2label = labeler.label(top_k_pairs)
        exclude_pairs.update(top_k_pairs)
        
        num_iter += 1
        
    ex_rdd.unpersist()
        
    

# map functions that apply to each partition
def create_executors(pairs_partition, bc_table_A, bc_table_B, bc_feature_info):
    pairs = [p for p in pairs_partition ]
    # executor node
	executor = cachenone.CacheNone()
	executor.prepare(bc_table_A.value, bc_table_B.value, bc_feature_info.value, pairs)
	executor.compute_features(list(range(len(bc_feature_info.value))), 
		bc_feature_info.value, bc_table_A.value, bc_table_B.value)
    return [executor]

def iteration(executors, rf, batch_size, exclude_pairs):
    combined = []
    for executor in executors:
        # apply random forest and select most informative examples
        top_k = executor.apply(rf, batch_size, exclude_pairs)
        combined.extend(top_k)
    return combined
	