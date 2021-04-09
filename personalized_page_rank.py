"""
Use Personalized Page Rank to perform recommendations.
TODO: for now this doesn't really take into account the ratings.
Best precision so far: 28% on 20 users with 5 recommendations each time (chosen among ~1000 animes)
"""

import networkx as nx
from tqdm import tqdm
import argparse

from utils import get_bipartite_graph, get_test


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
    G = get_bipartite_graph()
    test = get_test()

    good_recommendations, recommendations, watched = 0,0,0
    for user_id, anime_ids in tqdm(list(test.items())[:args.n_users]):
        recommended_anime_ids = recommend_ppr(G,user_id)
        
        good_recommendations += len(set(recommended_anime_ids).intersection(anime_ids))
        recommendations += len(recommended_anime_ids)
        watched += len(anime_ids)

    print(f"Precision on {args.n_users} users: {int(good_recommendations/recommendations*1000)/10}%")
    print(f"Recall: {int(good_recommendations/watched*1000)/10}%")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_users", type=int, default=10, help="number of users for validation")

    main(parser.parse_args())