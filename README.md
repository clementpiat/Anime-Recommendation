# Anime-Recommendation

You must first download the data on kaggle.
```
https://www.kaggle.com/CooperUnion/anime-recommendations-database/download
```

Then put the 2 `.csv` files in the data folder.
```
data/anime.csv
data/rating.csv
```

Then run this script to create `train.csv` and `test.csv`.
```
python split_train_test.py
```

Finally you can run the different experiments and test the models. For instance:
```
python personalized_page_rank.py -n 200
```

