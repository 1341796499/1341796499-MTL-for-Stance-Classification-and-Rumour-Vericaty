import os
import pandas as pd

def readProcessedCsv(all_csv_file_path: str) -> dict:
    all_csv_file_path_list = os.listdir(all_csv_file_path)

    for single_csv in all_csv_file_path_list:
        single_dataframe = pd.read_csv(os.path.join(all_csv_file_path, single_csv), dtype=object,lineterminator="\n")
        if single_csv == all_csv_file_path_list[0]:
            all_dataframe = single_dataframe
        else:
            all_dataframe = pd.concat([all_dataframe, single_dataframe], ignore_index=True)

    all_dataframe.rename(columns={"label_b\r": "label_b"}, inplace=True)
    all_dataframe["label_b"] = all_dataframe["label_b"].apply(lambda x: x.replace('\r', ''))
    all_dataframe["in_reply_to_status_id"].fillna(0, inplace=True)
    all_dataframe["in_reply_to_user_id_str"].fillna(0, inplace=True)
    all_dataframe["text"].fillna("null", inplace=True)
    label_a_convert = {
        "support": 0,
        "deny": 1,
        "query": 2,
        "comment": 3
    }
    label_b_convert = {
        "true": 0,
        "false": 1,
        "unverified": 2
    }
    all_dataframe["label_a"] = all_dataframe["label_a"].apply(lambda x: label_a_convert.get(x))
    all_dataframe["label_b"] = all_dataframe["label_b"].apply(lambda x: label_b_convert.get(x))
    all_dataframe["label_a"] = all_dataframe["label_a"].astype(int)
    all_dataframe["label_b"] = all_dataframe["label_b"].astype(int)
    remove_duplicated_df = all_dataframe.drop_duplicates(subset=["belong_to_which_source_tweet", "twitter_id"], keep="first").reset_index(drop=True)

    belong_to_which_source_tweet_map = {}

    for index, row in remove_duplicated_df.iterrows():
        item_dict = {
            "belong_to_which_source_tweet": row["belong_to_which_source_tweet"],
            "twitter_id": row["twitter_id"],
            "in_reply_to_status_id": row["in_reply_to_status_id"],
            "text": row["text"],
            "favorite_count": row["favorite_count"],
            "retweeted": row["retweeted"],
            "retweet_count": row["retweet_count"],
            "user_profile": {
                "id": row["user_id"],
                "verified": row["user_verified"],
                "followers_count": row["user_followers_count"],
                "listed_count": row["user_listed_count"],
                "statuses_count": row["user_statuses_count"],
                "description": row["user_description"],
                "friends_count": row["user_friends_count"],
                "location": row["user_location"],
                "following": row["user_following"],
                "geo_enabled": row["user_geo_enabled"],
                "name": row["user_name"],
                "favourites_count": row["user_favourites_count"],
                "screen_name": row["user_screen_name"],
                "created_at": row["user_created_at"],
                "time_zone": row["user_time_zone"]
            },
            "in_reply_to_user_id_str": row["in_reply_to_user_id_str"],
            "created_at": row["created_at"],
            "place": row["place"],
            "label_a": row["label_a"],
            "label_b": row["label_b"]
        }
        if row["belong_to_which_source_tweet"] not in belong_to_which_source_tweet_map:
            belong_to_which_source_tweet_map[row["belong_to_which_source_tweet"]] = {
                "label": row["label_b"],
                "item_dict_list": []
            }
            belong_to_which_source_tweet_map.get(row["belong_to_which_source_tweet"])["item_dict_list"].append(item_dict)
        else:
            belong_to_which_source_tweet_map.get(row["belong_to_which_source_tweet"])["item_dict_list"].append(item_dict)
    
    return belong_to_which_source_tweet_map
