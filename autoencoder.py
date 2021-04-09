import argparse
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from utils import get_bipartite_graph
from test import testing

class AutoEncoder(nn.Module):
    def __init__(self, K=5, dim_encoder=8, learning_rate=5e-5):
        super(AutoEncoder, self).__init__()
        self.MSELoss = nn.MSELoss()

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

        self.n = len(self.user_to_index)
        self.dim = len(self.anime_to_index)
        self.X = np.zeros((self.n, self.dim))

        for anime, user in self.G.edges():
            self.X[self.user_to_index[user], self.anime_to_index[anime]] = self.G[user][anime]['weight']

        self.X = torch.tensor(self.X, dtype=torch.float32)/10
        self.indices = list(range(self.n))

        self.encoder = nn.Sequential(
            nn.Linear(self.dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, dim_encoder),
            nn.ReLU(inplace=True),
            nn.Linear(dim_encoder, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.dim)
        )

        self.optimizer = Adam(self.parameters(), lr=learning_rate)

        self.fit()

    def fit(self):
        for epoch in range(1):
            print('\nEpoch', epoch+1)
            np.random.shuffle(self.indices)
            losses = []
            for index in tqdm(self.indices[:20000]):
                x = self.X[index]
                y = self.encoder(x)
                loss = self.MSELoss(y[x > 0], x[x > 0])

                losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if len(losses) == 1000:
                    print('Loss:', sum(losses)/1000)
                    losses = []

    def __call__(self, user_id):
        user_index = self.user_to_index[f"u_{user_id}"]

        scores = self.encoder(self.X[user_index]) * (self.X[user_index] == 0)
        
        movies_index = torch.topk(scores, self.K).indices.tolist()
        return [int(self.index_to_anime[i][2:]) for i in movies_index]

        

def main(args):
    testing(AutoEncoder(K=args.K), n_users=args.n_users)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_users", type=int, default=10, help="number of users for validation")
    parser.add_argument("--K", type=int, default=5, help="number of recommendations")

    main(parser.parse_args())