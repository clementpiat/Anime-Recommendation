"""
Best precision so far: 12.5% on 10 users with 5 recommendations each time (chosen among ~1000 animes)
"""

import networkx as nx
import argparse
import numpy as np
from utils import get_bipartite_graph

from test import testing

class KNN:
    def __init__(self, K=5, n_ngh=10):
        self.K = K
        self.n_ngh = n_ngh

        self.G = get_bipartite_graph()

        self.user_to_index = {}
        self.anime_to_index = {}

        for node in self.G.nodes():
            if node[0] == 'u':
                self.user_to_index[node] = len(self.user_to_index)
            else:
                self.anime_to_index[node] = len(self.anime_to_index)

        self.index_to_anime = {i: anime for (anime, i) in self.anime_to_index.items()}

        self.X = np.zeros((len(self.user_to_index), len(self.anime_to_index)))

        for anime, user in self.G.edges():
            self.X[self.user_to_index[user], self.anime_to_index[anime]] = self.G[user][anime]['weight']

    def __call__(self, user_id):
        user_index = self.user_to_index[f"u_{user_id}"]

        distances = (self.X @ self.X[user_index])/np.linalg.norm(self.X, axis=1)
        nghs = (-distances).argsort()[:self.n_ngh+1][1:]

        sum_rating = self.X[nghs].sum(axis=0)
        n_non_zeros = np.count_nonzero(self.X[nghs], axis=0)
        mean_ratings = np.divide(sum_rating, n_non_zeros, out=np.zeros_like(sum_rating), where=n_non_zeros!=0)

        movies_index = (-mean_ratings * (self.X[user_index] == 0)).argsort()[:self.K]
        return [int(self.index_to_anime[i][2:]) for i in movies_index]

        

def main(args):
    testing(KNN(K=args.K, n_ngh=args.n_ngh), n_users=args.n_users)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_users", type=int, default=10, help="number of users for validation")
    parser.add_argument("--n_ngh", type=int, default=10, help="number of neighbors")
    parser.add_argument("--K", type=int, default=5, help="number of recommendations")

    main(parser.parse_args())