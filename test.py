from tqdm import tqdm
import json
from collections import Counter

from utils import get_test

def testing(recommender, n_users=10):
    total_recommendations = []
    good_recommendations, recommendations, watched = 0,0,0

    test = get_test()
    for user_id, anime_ids in tqdm(list(test.items())[:n_users]):
        try:
            recommended_anime_ids = recommender(user_id)

            total_recommendations.extend(recommended_anime_ids)
            good_recommendations += len(set(recommended_anime_ids).intersection(anime_ids))
            recommendations += len(recommended_anime_ids)
            watched += len(anime_ids)
        except:
            # Unknown user
            pass
    
    print(json.dumps(Counter(total_recommendations), indent=2))
    print(f"\nPrecision on {n_users} users: {int(good_recommendations/recommendations*1000)/10}%")
    print(f"Recall: {int(good_recommendations/watched*1000)/10}%")