"""
Use Personalized Page Rank to perform recommendations.
TODO: for now this doesn't really take into account the ratings.
Best precision so far: 28% on 20 users with 5 recommendations each time (chosen among ~1000 animes)
"""

import networkx as nx
import argparse

from test import test

def recommend_ppr(G, user_id, K=5):
    try:
        ppr = nx.pagerank_scipy(G, alpha=0.7, personalization={f"u_{user_id}": 1})
        couples = sorted(ppr.items(), key=lambda x: 0 if G.has_edge(x[0], f"u_{user_id}") or x[0][0]=="u" else x[1])[-K:]
        return list(map(lambda x: int(x[0][2:]), couples))
    except:
        """
        If we didn't converge.
        This is bad for the recall but doesn't change the precision.
        """
        return []

def main(args):
    test(recommend_ppr, n_users=args.n_users)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_users", type=int, default=10, help="number of users for validation")

    main(parser.parse_args())