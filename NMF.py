"""
Best precision so far: 40.4% on 100 users with 5 recommendations each time (chosen among ~1000 animes)
"""

import networkx as nx
import argparse
import numpy as np
from sklearn.decomposition import NMF

from utils import get_bipartite_graph
from test import testing

class nmf:
    def __init__(self, K=5, n_components=10):
        self.K = K

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

        print('Computing NMF...')
        model = NMF(n_components=n_components, init='random', random_state=0)
        W = model.fit_transform(self.X)
        H = model.components_
        self.approx = W @ H
        print('Done\n')

        

    def __call__(self, user_id):
        user_index = self.user_to_index[f"u_{user_id}"]

        movies_index = (-self.approx[user_index] * (self.X[user_index] == 0)).argsort()[:self.K]
        return [int(self.index_to_anime[i][2:]) for i in movies_index]

        

def main(args):
    testing(nmf(K=args.K, n_components=args.n_components), n_users=args.n_users)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_users", type=int, default=10, help="number of users for validation")
    parser.add_argument("--n_components", type=int, default=10, help="number of components")
    parser.add_argument("--K", type=int, default=5, help="number of recommendations")

    main(parser.parse_args())