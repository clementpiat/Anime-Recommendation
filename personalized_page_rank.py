"""
Use Personalized Page Rank to perform recommendations.
TODO: for now this doesn't really take into account the ratings.
Best precision so far: 28% on 20 users with 5 recommendations each time (chosen among ~1000 animes)
"""

import networkx as nx
import argparse
from fast_pagerank import pagerank_power
import numpy as np
from utils import get_bipartite_graph

from test import testing

class PPR:
    def __init__(self, K=5):
        self.adj_matrix, self.nodes = get_bipartite_graph(as_scipy=True)
        self.nodes_to_index = dict([(node,i) for i,node in enumerate(self.nodes)])
        self.K = K

    def __call__(self, user_id):
        user_index = self.nodes_to_index[f"u_{user_id}"]

        personalize = np.zeros(self.adj_matrix.shape[0])
        personalize[user_index] = 1
        ppr = pagerank_power(self.adj_matrix, p=0.6, personalize=personalize, tol=1e-6)
        couples = sorted(enumerate(ppr), key=lambda x: 0 if self.adj_matrix[user_index, x[0]] or self.nodes[x[0]][0]=="u" else x[1])[-self.K:]
        return list(map(lambda x: int(self.nodes[x[0]][2:]), couples))

def main(args):
    testing(PPR(), n_users=args.n_users)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_users", type=int, default=10, help="number of users for validation")

    main(parser.parse_args())