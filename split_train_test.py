import pandas as pd
from sklearn.model_selection import train_test_split

def main(test_size=0.3, min_n_animes=30, min_n_users=3000, min_rating=6):
    """
    min_n_animes: min number of animes that a user have watched
    min_n_users: min number of users/watchers for an anime 
    min_rating: minimum rating for considering someone "liked" an anime
    In case of memory issues, consider augmenting min_n_users to reduce the number of animes in the graph
    """
    anime = pd.read_csv("data/anime.csv")
    rating = pd.read_csv("data/rating.csv")

    # Filtering rating
    rating = rating[rating.rating >= min_rating]

    # Filtering animes
    grouped_rating = rating.groupby("anime_id").count()
    anime_ids = list(grouped_rating[grouped_rating["user_id"]>=min_n_users].index)
    filtered_rating = rating[rating["anime_id"].isin(anime_ids)]

    # Filtering users
    grouped_filtered_rating = filtered_rating.groupby("user_id").count()
    user_ids = list(grouped_filtered_rating[grouped_filtered_rating["anime_id"]>=min_n_animes].index)
    filtered_rating = filtered_rating[filtered_rating["user_id"].isin(user_ids)]

    # Split
    train_index, test_index = train_test_split(list(filtered_rating.index), test_size=test_size)
    train = filtered_rating[filtered_rating.index.isin(train_index)]
    test = filtered_rating[filtered_rating.index.isin(test_index)]

    print("Saving datasets...")
    train.to_csv("data/train_rating.csv", index=False)
    test.to_csv("data/test_rating.csv", index=False)



if __name__ == "__main__":
    main()