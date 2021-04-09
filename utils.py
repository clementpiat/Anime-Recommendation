import pandas as pd
from tqdm import tqdm
import numpy as np
import networkx as nx
from scipy import sparse

def sparsity(rating):
    n_ratings = len(rating)
    n_animes = len(rating.anime_id.unique())
    n_users = len(rating.user_id.unique())

    return 1 - n_ratings/(n_animes*n_users)
    

def get_bipartite_graph(min_rating=6, sparse=True):
    """
    Returns a networkx bipartite graph
    min_rating: minimum rating for considering someone "liked" an anime
    """
    rating = pd.read_csv("data/train_rating.csv")

    print(f"Number of animes: {len(rating.anime_id.unique())}")
    print(f"Number of users: {len(rating.user_id.unique())}")
    print(f"Sparsity: {sparsity(rating)}")
    
    # Build the bipartite graph
    G = nx.Graph()
    for i in rating.anime_id.unique():
        G.add_node("a_" + str(i))

    for row in tqdm(np.array(rating)):
        if row[2] >= min_rating:
            u_node = "u_" + str(row[0])
            a_node = "a_" + str(row[1])
            G.add_edge(u_node, a_node, weight=row[2])

    return G

def get_test():
    """
    Returns a dict in the following format:
    {   
        <user1>: [<anime3>,<anime4>,<anime7>]
        <user2>: [<anime1>, ...]
        ...
    }
    """
    rating = pd.read_csv("data/test_rating.csv")
    return rating.groupby("user_id")["anime_id"].apply(list).to_dict()