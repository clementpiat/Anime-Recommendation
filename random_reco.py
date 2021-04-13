"""
Best precision so far: 40.4% on 100 users with 5 recommendations each time (chosen among ~1000 animes)
"""

import argparse
import numpy as np

from utils import get_bipartite_graph
from test import testing

class RandomRecommender:
    def __init__(self, K=5):
        self.K = K

        self.G = get_bipartite_graph()

        self.anime_ids = set()

        for node in self.G.nodes():
            if node[0] == 'a':
                self.anime_ids.add(node)
        

    def __call__(self, user_id):
        user = f"u_{user_id}"

        nghs = set(self.G.neighbors(user))
        choices = list(self.anime_ids - nghs)
        return [int(x[2:]) for x in np.random.choice(choices, size=self.K, replace=False)]

        

def main(args):
    testing(RandomRecommender(K=args.K), n_users=args.n_users)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_users", type=int, default=200, help="number of users for validation")
    parser.add_argument("--K", type=int, default=5, help="number of recommendations")

    main(parser.parse_args())