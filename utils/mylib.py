import pandas as pd
import numpy as np

DATA_PATH = "C:/Users/lenovo/Desktop/github/machineLearning/data/ml-1m"
ratings = pd.read_csv(
    f"{DATA_PATH}/ratings.dat",
    sep="::",
    engine="python",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

movies = pd.read_csv(
    f"{DATA_PATH}/movies.dat",
    sep="::",
    engine="python",
    encoding="latin1",
    names=["movie_id", "title", "genres"]
)

users = pd.read_csv(
    f"{DATA_PATH}/users.dat",
    sep="::",
    engine="python",
    names=["user_id", "gender", "age", "occupation", "zipcode"]
)

def get_data():
    return ratings, movies, users