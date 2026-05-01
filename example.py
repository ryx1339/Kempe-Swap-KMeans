import pandas as pd
import numpy as np
from src.KSKM import KSKM
from sklearn.metrics import adjusted_rand_score

# An illustrative example from paper (https://arxiv.org/pdf/2603.27417)

# Define data
data = [[0, 0, 0],
        [0, 2, 0],
        [0, 4, 0],
        [0, 6, 1],
        [2, 0, 0],
        [2, 2, 0],
        [2, 4, 0],
        [2, 6, 1],
        [4, 0, 3],
        [4, 2, 2],
        [4, 4, 2],
        [4, 6, 2],
        [6, 0, 3],
        [6, 2, 2],
        [6, 4, 2],
        [6, 6, 2]]

df = pd.DataFrame(data, columns=['x1', 'x2', 'ground_truth_label'])

# Define problem input parameters and sets
# number of cluster
k = df['ground_truth_label'].unique().size
data = df[['x1', 'x2']].values
true_label = df['ground_truth_label'].values

#  Define information on pairs of objects
# hard_must_link_constraints
ml = [(2, 6), (9, 10)]
#hard_cannot_link_constraints 
cl = [(2, 7), (6, 7), (5, 11)]


#  Run algorithm with parameters
steps_back_to_best = 5
steps_no_improvement = 10
steps_mutation = 200
verbose = False
random_state = 603
np.random.seed(random_state)

membership = KSKM(random_state, data, ml, cl, steps_mutation, k, steps_back_to_best, steps_no_improvement, verbose)

# Evaluate assignment
print('ARI: ', adjusted_rand_score(true_label, membership))