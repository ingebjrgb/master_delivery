import math
import os
import numpy as np


class CollaborateFilter:
    def __init__(self, input_file_name, k):
        self.input_file_name = input_file_name
        self.k = k

        self.useritem_matrix = None
        self.uu_dataset = None

        self.generated_ratings = np.full((845, 1574), 0, dtype=float)
        self.usersimilarities = np.full((845, 845), np.nan, dtype=float)

    def initialize(self):
        """
        Initialize and check parameters
        """

        # check file exist and if it's a file or dir
        if not os.path.isfile(self.input_file_name):
            self.quit("Input file doesn't exist or it's not a file")

        # load data
        self.useritem_matrix, self.uu_dataset = self.load_data(self.input_file_name)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """                             Pearson Correlation                              """
    """                                                                              """
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def pearson_correlation(self, user1,
                            user2):
        user1_data = self.uu_dataset[user1]
        user2_data = self.uu_dataset[user2]

        rx_avg = self.user_average_rating(user1_data)
        ry_avg = self.user_average_rating(user2_data)
        sxy = self.common_items(user1_data, user2_data)
        if len(sxy) == 0:
            return 0

        top_result = 0.0
        bottom_left_result = 0.0
        bottom_right_result = 0.0
        for item in sxy:
            rxs = user1_data[item]
            rys = user2_data[item]
            top_result += (rxs - rx_avg) * (rys - ry_avg)
            bottom_left_result += pow((rxs - rx_avg), 2)
            bottom_right_result += pow((rys - ry_avg), 2)
        bottom_left_result = math.sqrt(bottom_left_result)
        bottom_right_result = math.sqrt(bottom_right_result)

        if bottom_left_result == 0:
            return 0
        if bottom_right_result == 0:
            return 0

        result = top_result / (bottom_left_result * bottom_right_result)
        return result

    def inverse_pearson_correlation(self, user1, user2):
        user1_data = self.uu_dataset[user1]
        user2_data = self.uu_dataset[user2]

        # we want to invert the rating of user 2,  r' = r_max − r + r_min
        user2_data = {movie: int(5 - rating + 1) for (movie, rating) in user2_data.items()}

        rx_avg = self.user_average_rating(user1_data)
        ry_avg = self.user_average_rating(user2_data)

        sxy = self.common_items(user1_data, user2_data)

        # minimum co-rated items constraint is set to 3
        if len(sxy) < 3:
            return 0

        top_result = 0.0
        bottom_left_result = 0.0
        bottom_right_result = 0.0
        for item in sxy:
            rxs = user1_data[item]
            rys = user2_data[item]
            top_result += (rxs - rx_avg) * (rys - ry_avg)
            bottom_left_result += pow((rxs - rx_avg), 2)
            bottom_right_result += pow((rys - ry_avg), 2)
        bottom_left_result = math.sqrt(bottom_left_result)
        bottom_right_result = math.sqrt(bottom_right_result)

        if bottom_left_result == 0:
            return 0
        if bottom_right_result == 0:
            return 0

        result = top_result / (bottom_left_result * bottom_right_result)
        return result

    def user_average_rating(self, user_data):
        avg_rating = 0.0
        size = len(user_data)
        for (movie, rating) in user_data.items():
            avg_rating += float(rating)
        avg_rating /= size * 1.0
        return avg_rating

    def common_items(self, user1_data, user2_data):
        result = []
        ht = {}
        for (movie, rating) in user1_data.items():
            ht.setdefault(movie, 0)
            ht[movie] += 1
        for (movie, rating) in user2_data.items():
            ht.setdefault(movie, 0)
            ht[movie] += 1
        for (k, v) in ht.items():
            if v == 2:
                result.append(k)
        return result

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """                             K Nearest Neighbors                              """
    """                                                                              """
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def k_nearest_neighbors(self, user):
        neighbors = []
        result = []
        for (user_id, data) in self.uu_dataset.items():
            if user_id == user:
                continue
            if np.isnan(self.usersimilarities[
                            user_id - 1, user - 1]):
                # upc = self.pearson_correlation(user, user_id)
                upc = self.inverse_pearson_correlation(user, user_id)
                neighbors.append([user_id, upc])
                self.usersimilarities[user_id - 1, user - 1] = upc
                self.usersimilarities[user - 1, user_id - 1] = upc

            else:
                neighbors.append([user_id, self.usersimilarities[user_id - 1, user - 1]])

        sorted_neighbors = sorted(neighbors, key=lambda neighbors: (neighbors[1], neighbors[0]),
                                  reverse=True)  # - for desc sort

        for i in range(self.k):
            if i >= len(sorted_neighbors):
                break
            result.append(sorted_neighbors[i])
        return result

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """                                  Predict                                     """
    """                                                                              """
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def predict(self, user, k_nearest_neighbors):
        for item_index, rating in enumerate(self.useritem_matrix[user - 1]):
            if rating == 0:
                result = self.predict_item(item_index + 1, k_nearest_neighbors)
                self.generated_ratings[user - 1, item_index] = result
        return self.generated_ratings[user - 1]

    def predict_item(self, item, k_nearest_neighbors):
        valid_neighbors = self.check_neighbors_validattion(item, k_nearest_neighbors)
        if not len(valid_neighbors):
            return 0.0
        top_result = 0.0
        bottom_result = 0.0
        for neighbor in valid_neighbors:
            neighbor_id = neighbor[0]
            neighbor_similarity = neighbor[1]  # Wi1
            # rating = self.uu_dataset[neighbor_id][item]  # rating i,item
            rating = 5 - self.uu_dataset[neighbor_id][item] + 1  # rating i,item for inverse
            top_result += neighbor_similarity * rating
            bottom_result += neighbor_similarity
        result = top_result / bottom_result
        return result

    def check_neighbors_validattion(self, item, k_nearest_neighbors):
        result = []
        for neighbor in k_nearest_neighbors:
            neighbor_id = neighbor[0]
            similarity = neighbor[1]
            # print item
            if item in self.uu_dataset[neighbor_id].keys() and similarity != 0:
                result.append(neighbor)
        return result

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """                             Helper Functions                                 """
    """                                                                              """
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def load_data(self, input_file_name):
        """
        load data and return two outputs for extention purpose
        """
        input_file = open(input_file_name, 'r')
        useritem_matrix = np.zeros(shape=(845, 1574))
        uu_dataset = {}
        for line in input_file:
            userid, itemid, rating, timestamp = line.split("\t")
            userid, itemid, rating = int(userid), int(itemid), int(rating)
            useritem_matrix[userid - 1, itemid - 1] = rating

            """
            user-user dataset: {user: [0: Movie Name  1: Rating]}
            """
            uu_dataset.setdefault(userid, {})
            uu_dataset[userid].setdefault(itemid, float(rating))

        return useritem_matrix, uu_dataset

    def get_top_10(self, user):
        """
        return top 10 items for user (returns the real itemID)
        """
        sorted_ix = np.argsort(self.generated_ratings[user - 1])
        sorted_ix += 1  # to make up for 1 indexing of the itemIDs
        sorted_ix = np.flip(sorted_ix, 0)  # make sure that list is sorted from top item to worst item
        return sorted_ix[:10]

    def write_to_file(self):
        """
        Writes the recommendation data to file where each new line consists of "userID  movieID" for each rec
        """
        with open(
                "kfn_recs/recs.tsv",
                'w') as f:
            content = []
            for user_index in range(845):
                userID = user_index + 1
                for itemID in self.get_top_10(userID):
                    content.append(f"{userID}\t{itemID}\n")
            f.writelines(content)

    def quit(self, err_desc):
        tips = "\n" + "TIPS: " + "\n" \
               + "--------------------------------------------------------" + "\n" \
               + "Pragram name: lingzhe_teng_collabFilter.py" + "\n" \
               + "First parameter: Input File, e.g. ratings-dataset.tsv" + "\n" \
               + "Second parameter: K, e.g. 10" + "\n" \
               + "--------------------------------------------------------" + "\n" \
               + "Note:" + "\n" \
               + "Please use double quotation marks, such as \"USER\'S ID\" or \"MOVIEW\'S NAME\", for User ID and Moview parameters" + "\n"

        raise SystemExit('\n' + "PROGRAM EXIT: " + err_desc + ', please check your input' + '\n' + tips)


if __name__ == '__main__':
    input_file_name = '/Users/ingebjorgbarthold/projects/master_recommendation_blur/splits/train_n_val_n_dislikes.tsv'
    user = 3
    k = 10  # k neighbors

    cf = CollaborateFilter(input_file_name, k)
    cf.initialize()

    # k_nearest_neighbors = cf.k_nearest_neighbors(user)
    # prediction = cf.predict(user, k_nearest_neighbors)
    # print("top items", cf.get_top_10(user))  # denne viser top-items
    # print("score for top items", cf.generated_ratings[user-1, cf.get_top_10(user)-1])
    # print("overall scores for user", cf.generated_ratings[user-1].tolist())

    for user in range(1, 846):
        k_nearest_neighbors = cf.k_nearest_neighbors(user)

        prediction = cf.predict(user,
                                k_nearest_neighbors)
    cf.write_to_file()
