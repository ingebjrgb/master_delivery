import pandas as pd
import numpy as np


def recommender_file_to_df(file, src_len):
    """
    Returns a dataframe from the recommendation file.

            Parameters:
                    file (string): the filename of a file in the folder 'rec_output'
                    src_len (int): number of recommendations in each recommender list

            Returns:
                    dataframe with columns [userID, i0, i1, i2, i3,... ik-1]
    """
    recommendations = [[i] for i in range(
        0, 846)]
    cols = [f'i{i}' for i in range(src_len)]
    cols.insert(0, 'userID')
    with open(f'{file}') as f:
        for line in f.readlines():
            userID, movieID = line.split()
            recommendations[int(userID)].append(int(movieID))
    df_rec = pd.DataFrame(recommendations[1:], columns=cols) # skip first because there are no userID = 0
    return df_rec


def load_gender_vector():
    """
    Loads and returns the gender for all users

        Returns:
             the gender vector. For each index (corresponding to the userID-1), a male is 0 and female 1
        """
    gender_vec = []
    with open("preprocessed/ml100k_users_detailed_Areas.csv", 'r') as f:
        for line in f.readlines()[1:]:
            user_id, age, gender, occ, postcode = line.split(",")[0:5]
            if gender == "M":
                gender_vec.append(0)
            else:
                gender_vec.append(1)
    return np.asarray(gender_vec)


def rating_file_to_df():
    """
            Returns:
                    a dataframe containing the information from all ratings in the train and validation set
                    with the columns [UserID, MovieID, Rating, Timestamp] and a row for all the recorded ratings
    """
    columns_rating = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    df_rating_train = pd.read_csv('./splits/train.tsv', sep='\t',
                                  names=columns_rating, header=None, engine='python')
    df_rating_val = pd.read_csv('./splits/val.tsv', sep='\t',
                                names=columns_rating, header=None, engine='python')
    df_rating = pd.concat([df_rating_train, df_rating_val], ignore_index=True)

    return df_rating


def read_gender_indicative_file():
    """
            Returns:
                    two lists of all movieIDs (as strings) that are typically male and female, respectively
    """
    gender_indicative_movies = [0, 0]
    with open('gender_indicative_movies.txt') as f:
        for gender, line in enumerate(f.readlines()):
            title, items = line.strip('\n').split(":")
            gender_indicative_movies[gender] = items.split(',')
    L_m = gender_indicative_movies[0]
    L_f = gender_indicative_movies[1]
    return L_m, L_f


def load_recs_as_user_item_matrix(file='gender_sideinfo/top10_WARP_R2.tsv', max_user=845, max_item=1574):
    """
    Loads the recommendations in the format of a user-item matrix R from the generated recommendations. Useful in the
    inference attack scenario

        Parameters:
            file(string): the path of the recommendation list file
            max_user (int): a threshold for the maximum userID
            max_item (int): a threshold for the maximum movieID

        Returns:
            A user-item matrix as ndarray
    """
    R = np.zeros(shape=(max_user, max_item))
    with open(f'./rec_output/{file}') as f:
        for line in f.readlines():
            user, item = line.split()
            R[int(user) - 1, int(item) - 1] = 1
    return R

