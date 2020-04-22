'''
What you need to code:
1. Create your own seeder.py, which simply returns seeding pairs
2. Create your own labeler.py, which can either simulate labeling using golden labels, or do manual labeling 

What need to be done before calling baseline.run
1. Loading tables A and B of entities into Pandas dataframes (table_A, table_B)
2. Loading candidate pairs (each pair is a pair of entity ids from tables A and B (candidate_pairs)
3. Loading golden labels of those candidate pairs (table_G)
4. Get the tables of features between tables A and B using py_entitymatching (feature_table)
5. Note that feature_info is a subset of columns from feature_table
6. create the SparkContext (sc)
7. Specify params like number of executors (num_executors), number of seeding pairs (seed_size), 
  max number of iterations (max_iter) and number of pairs to label per iteration (batch_size)
 
Now you can pass them to baseline.run, which will simulate active learning on between A,B for EM
'''
