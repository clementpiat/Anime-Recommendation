"""
Use Personalized Page Rank to perform recommendations.
TODO: for now this doesn't really take into account the ratings.
Best precision so far: 28% on 20 users with 5 recommendations each time (chosen among ~1000 animes)
"""

import networkx as nx
from tqdm import tqdm
import argparse
from fast_pagerank import pagerank_power
import numpy as np
from collections import Counter
import json

from utils import get_bipartite_graph, get_test


def recommend_ppr(adj_matrix, nodes, user_index, K=5):

    personalize = np.zeros(adj_matrix.shape[0])
    personalize[user_index] = 1
    ppr = pagerank_power(adj_matrix, p=0.6, personalize=personalize, tol=1e-6)
    couples = sorted(enumerate(ppr), key=lambda x: 0 if adj_matrix[user_index, x[0]] or nodes[x[0]][0]=="u" else x[1])[-K:]
    return list(map(lambda x: int(nodes[x[0]][2:]), couples))
    
    # ppr = nx.pagerank_scipy(G, alpha=0.7, personalization={f"u_{user_id}": 1})
    # couples = sorted(ppr.items(), key=lambda x: 0 if G.has_edge(x[0], f"u_{user_id}") or x[0][0]=="u" else x[1])[-K:]
    # return list(map(lambda x: int(x[0][2:]), couples))

def main(args):
    adj_matrix, nodes = get_bipartite_graph(as_scipy=True)
    nodes_to_index = dict([(node,i) for i,node in enumerate(nodes)])
    test = get_test()

    total_recommendations = []
    good_recommendations, recommendations, watched, n_users = 0,0,0,0
    for user_id, anime_ids in tqdm(list(test.items())[:args.n_users]):
        try:
            user_index = nodes_to_index[f"u_{user_id}"]
            recommended_anime_ids = recommend_ppr(adj_matrix, nodes, user_index)

            total_recommendations.extend(recommended_anime_ids)
            good_recommendations += len(set(recommended_anime_ids).intersection(anime_ids))
            recommendations += len(recommended_anime_ids)
            watched += len(anime_ids)
            n_users += 1
        except:
            # Unknown user
            pass
    
    print(json.dumps(Counter(total_recommendations), indent=2))
    print(f"\nPrecision on {n_users} users: {int(good_recommendations/recommendations*1000)/10}%")
    print(f"Recall: {int(good_recommendations/watched*1000)/10}%")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_users", type=int, default=10, help="number of users for validation")

    main(parser.parse_args())