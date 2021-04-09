from tqdm import tqdm
import json
from collections import Counter

from utils import get_bipartite_graph, get_test

def test(recommender, n_users=10):
    adj_matrix, nodes = get_bipartite_graph(as_scipy=True)
    nodes_to_index = dict([(node,i) for i,node in enumerate(nodes)])
    test = get_test()

    total_recommendations = []
    good_recommendations, recommendations, watched, n_users = 0,0,0,0
    for user_id, anime_ids in tqdm(list(test.items())[:n_users]):
        try:
            user_index = nodes_to_index[f"u_{user_id}"]
            recommended_anime_ids = recommender(adj_matrix, nodes, user_index)

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