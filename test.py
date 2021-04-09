from tqdm import tqdm

from utils import get_bipartite_graph, get_test

def test(recommender, n_users=10, G=None, test=None):
    if G is None:
        G = get_bipartite_graph()
    if test is None:
        test = get_test()

    good_recommendations, recommendations, watched = 0,0,0
    for user_id, anime_ids in tqdm(list(test.items())[:n_users]):
        recommended_anime_ids = recommender(G,user_id)
        
        good_recommendations += len(set(recommended_anime_ids).intersection(anime_ids))
        recommendations += len(recommended_anime_ids)
        watched += len(anime_ids)

    print(f"Precision on {n_users} users: {int(good_recommendations/recommendations*1000)/10}%")
    print(f"Recall: {int(good_recommendations/watched*1000)/10}%")