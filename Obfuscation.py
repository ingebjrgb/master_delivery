from numpy.random import default_rng
import numpy as np
from collections import Counter
from utils import recommender_file_to_df, rating_file_to_df, read_gender_indicative_file, load_gender_vector


def main(**kwargs):
    global L_m, L_f  # items that are most correlated to males and females, respectively
    L_m, L_f = read_gender_indicative_file()
    global gender_vec  # the gender vector where gender_vec[userID-1] denotes userIDs gender
    gender_vec = load_gender_vector()

    # variables
    src_len = 10  # length of recommender list
    rec_file = './rec_output/gender_sideinfo/top10_WARP_R2.tsv'  # gender_sideinfo/top10_WARP_R2.tsv or gender_sideinfo/WARP_R2.tsv
    removal_type = "kfn"  # options: gender, gender-random, random, kfn
    insertion_type = "kfn-addbehind"  # options: popularity, random, popularity-addbehind, random-addbehind, kfn-addbehind
    proportions = [0.2, 0.4, 0.6]  # the proportion of items to be replaced
    rng = default_rng(seed=0)  # random generator

    df_his_ratings = rating_file_to_df()
    df_recommendations = recommender_file_to_df(rec_file, src_len)
    for proportion in proportions:
        obfuscated_df = protection(removal_type, insertion_type, src_len,
                                   proportion, df_recommendations, df_his_ratings, rng)

        # Saving the obfuscation to file so that we can use Elliot and pycaret for performance measuring and inference
        # attack
        print_obfuscated_recs_to_file(obfuscated_df, removal_type, insertion_type, proportion)


def print_obfuscated_recs_to_file(obfuscated_df, removal_type, insertion_type, proportion):
    """
        Prints the recommendation data to file where each new line consists of "userID  movieID" for each rec

                Parameters:
                        obfuscated_df (DataFrame): on the format [userID, i0, i1,i2,...ik-1]
                        removal_type (String): the removal type used in this obfuscation
                        insertion_type (String): the insertion type used in this obfuscation
                        proportion (float): the proportion of items to be removed type used in this obfuscation
        """
    with open(f"rec_output/obfuscated/obfuscated_recs_r_{removal_type}_i_{insertion_type}_prop_{proportion}.tsv", 'w+') as f:
        content = []
        for row_ix, row in obfuscated_df.iterrows():
            userID = row.iloc[0]
            for itemID in row.iloc[1:].values.tolist():
                content.append(f"{userID}\t{itemID}\n")
        f.writelines(content)


def create_complete_itemset(df):
    """
    Finds all items that are rated and how popular they are

            Parameters:
                    df (DataFrame): the historical ratings of all users

            Returns: a list with all the items along with a corresponding list of the percentage of how often each
            item is interacted with (popularity)
    """
    all_items = df['MovieID'].values.tolist()
    itemcounter = Counter(all_items)
    all_items = list(itemcounter)
    tot_count = sum((itemcounter[item] for item in all_items))
    probabilities = [itemcounter[item] /
                     tot_count for item in all_items]
    return all_items, probabilities


def replace(replace_num, src_len, replace_type, rec_list, gender, rng):
    """
    Chooses which of the items in the recommender list to remove

            Parameters:
                    replace_num (int): the number of items to be removed
                    src_len(int): lenght of recommender list
                    replace_type (string): the strategy for choosing items to be replaced
                    rec_list (list): the items recommended
                    gender (int): 0 for male, 1 for female

            Returns:
                    a list of length replace_num with the index of all items to be replaced (removed)
    """
    if replace_type == 'random':
        # return a list of randomly unique integers of the length of replace_num that symbolizes the items to be
        # replaced
        # rng = default_rng()
        replace_positions = rng.choice(src_len, replace_num, replace=False)
        return replace_positions

    elif 'gender' in replace_type:
        # return a list of index for the positions in the  recommender list that are most gender indicative and
        # select eventual further positions randomly
        replace_items = []
        # rng = default_rng()

        if gender == 0:  # male
            gender_indicative_reclist = [int(x) for x in L_m if int(x) in rec_list]
        elif gender == 1:  # female
            gender_indicative_reclist = [int(x) for x in L_f if int(x) in rec_list]
        else:
            return []

        gender_indicative_iterator = iter(gender_indicative_reclist)
        while len(replace_items) < replace_num:
            # Choose the most gender indicative items until there are no more gender indicative items in the rec_list
            replace_items.append(
                next(gender_indicative_iterator, None))
        replace_positions = [rec_list.index(
            item) if item is not None else None for item in replace_items]  # Map each item to its index in the rec_list

        # for the instance when we want to select random positions for the remaining items in regard to replace_num
        if replace_type == 'gender-random':
            # If we have one or more instances of None in the replace_positions list
            if None in replace_positions:
                # create a list of the unchosen positions of the rec_list
                unchosen_ix = list(
                    filter(lambda ix: ix not in replace_positions, np.arange(src_len)))
                # Find index of first None
                index = replace_positions.index(None)
                # Replace all items after first None (which are also None) with random positions (indexes) of the
                # rec_list
                replace_positions[index:] = rng.choice(
                    unchosen_ix, replace_num - index, replace=False)
        return replace_positions
    elif replace_type == 'kfn':
        return [pos for pos in range(src_len)[-replace_num:]] # the last 'replace_num' number of items
    else:
        return []


def protection(removal_type, insertion_type, src_len, proportion, df_rec, df_his, rng):
    """
    Implements the replacement of items (removal and insertion)

            Parameters:
                    removal_type (string): the strategy for choosing items to insert
                    insertion_type (string): the strategy for choosing items to remove
                    src_len(int): length of recommendation list
                    proportion (float): the proportion of items to be replaced
                    df_rec (DataFrame): the recommendations generated for each user
                    df_his (DataFrame): the historical ratings for all users

            Returns:
                    a dataframe with the new obfuscated recommendations with columns userID, i0, i1, i2, i3,...ik-1
    """
    # number of items to be replaced
    replace_num = int(np.ceil(src_len * proportion))

    obfuscated_dataframe = df_rec.copy()
    all_items, probabilities = create_complete_itemset(df_his)

    if 'kfn' in insertion_type:
        kfn_df = recommender_file_to_df(
            '/Users/ingebjorgbarthold/projects/master_recommendation_blur/kfn_recs/recs.tsv', src_len)

    # Iterate through all users
    for user_index, row in df_rec.iterrows():
        # recommender list for the given user
        user_recs = row.iloc[1:].values.tolist()

        # userID corresponds to the userID, and user_index is the index (zero indexed) for the corresponding
        # dataframe row
        user_gender = gender_vec[user_index]

        # create a list of all items not including items that are already in the recommender list for the given user
        # and create a corresponding probabilities list
        displayed_items_ix = [all_items.index(
            int(shown_item)) for shown_item in user_recs]
        choosable_items = [item for index, item in enumerate(
            all_items) if index not in displayed_items_ix]
        choosable_prob = [prob for index, prob in enumerate(
            probabilities) if index not in displayed_items_ix]
        choosable_prob = np.array(choosable_prob)
        choosable_prob = choosable_prob / (sum(choosable_prob))

        if 'random' in insertion_type:
            # choose items to insert
            insertion_ids = rng.choice(
                choosable_items, replace_num, replace=False)
        elif 'popularity' in insertion_type:
            insertion_ids = rng.choice(
                choosable_items, replace_num, p=choosable_prob, replace=False)
        elif 'kfn' in insertion_type:
            insertion_ids = np.zeros(replace_num) # added to pass the assertion test
            kfn_recs = kfn_df.iloc[user_index].iloc[1:].values.tolist()
        else:
            insertion_ids = []

        # find positions to replace
        removal_positions = replace(
            replace_num, src_len, removal_type, user_recs, user_gender, rng)

        # ensure that the number of items to be inserted is the same as items that are replaced
        assert len(removal_positions) == replace_num, "The list of postions to be replaced is to short. Problem with " \
                                                      "the list removal_positions."
        assert len(insertion_ids) == replace_num, "The list of items to be inserted is to short. Problem with the " \
                                                  "list insertion_ids."

        if 'addbehind' in insertion_type: # If we want to add the new items at the end of the list
            new_recs = user_recs.copy()
            new_recs = [rec for plc, rec in enumerate(new_recs) if plc not in removal_positions]
            if 'kfn' in insertion_type:
                insertion_ids = [item for item in kfn_recs if item not in new_recs]
            new_recs.extend(insertion_ids)
            new_recs = new_recs[0:10]
            assert len(new_recs) == src_len, "The updated list of recommended items are not the same length as the " \
                                             "original recommender list"
            for ix, new_item in enumerate(new_recs):
                obfuscated_dataframe.at[user_index, f'i{ix}'] = new_item

        else:
            for j, removal_pos in enumerate(removal_positions):
                if removal_pos is None:
                    continue
                # insert the selected items to the positions that are to be replaced
                obfuscated_dataframe.at[user_index, f'i{removal_pos}'] = insertion_ids[j]

    print_info = {
        "Insertion strategy ": insertion_type,
        "Removal Strategy": removal_type,
        "Number of items to be replaced": replace_num,
    }
    print(print_info)
    return obfuscated_dataframe


if __name__ == "__main__":
    main()
