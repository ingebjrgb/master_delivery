from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import numpy as np
import scipy.stats as ss
from utils import load_gender_vector


def load_user_item_matrix(max_user=845, max_item=1574):
    """
    Loads the user-item matrix R from the complete (preprocessed) movielens100k data set.

        Parameters:
            max_user (int): a threshold for the maximum userID
            max_item (int): a threshold for the maximum movieID

        Returns:
            A user-item matrix as ndarray
    """
    R = np.zeros(shape=(max_user, max_item))
    with open("splits/train.tsv", 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("\t")
            user_id, movie_id, rating = int(
                user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item:
                R[user_id - 1, movie_id - 1] = rating
    with open("splits/val.tsv", 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("\t")
            user_id, movie_id, rating = int(
                user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item:
                R[user_id - 1, movie_id - 1] = rating
    with open("splits/test.tsv", 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("\t")
            user_id, movie_id, rating = int(
                user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item:
                R[user_id - 1, movie_id - 1] = rating
    return R


X = load_user_item_matrix()
T = load_gender_vector()
top = -1

# Get the set of most correlated movies, L_f and L_m:

cv = StratifiedKFold(n_splits=10)
coefs = []
avg_coefs = np.zeros(shape=(len(X[1]),))

random_state = np.random.RandomState(0)
for train, test in cv.split(X, T):
    x, t = X[train], T[train]
    model = LogisticRegression(
        penalty='l2', random_state=random_state, max_iter=500)

    model.fit(x, t)
    # rank the coefs:
    ranks = ss.rankdata(model.coef_[0])
    coefs.append(ranks)
    avg_coefs += model.coef_[0]

coefs = np.average(coefs, axis=0)
coefs = [[coefs[i], i + 1, avg_coefs[i]] for i in range(len(coefs))]
coefs = np.asarray(list(sorted(coefs)))

if top == -1:
    values = coefs[:, 2]
    var_val = np.min(np.abs(values))
    index_zero = np.where(np.abs(values) == np.min(np.abs(values)))
    top_male = index_zero[0][0]
    top_female = index_zero[0][-1]
    L_m = coefs[:top_male, 1]
    L_f = coefs[coefs.shape[0] - top_female:, 1]
    L_f = list(reversed(L_f))

else:
    L_m = coefs[:top, 1]
    L_f = coefs[coefs.shape[0] - top:, 1]
    L_f = list(reversed(L_f))

L_m = [int(x) for x in L_m]
L_f = [int(x) for x in L_f]

with open("gender_indicative_movies.txt", 'w+') as f:
    L_f_string = ''.join(str(item) + ',' if index < len(L_f) -
                                            1 else str(item) for index, item in enumerate(L_f))
    L_m_string = ''.join(str(item) + ',' if index < len(L_m) -
                                            1 else str(item) for index, item in enumerate(L_m))

    content = "L_m:" + L_m_string + "\nL_f:" + L_f_string
    f.write(content)
