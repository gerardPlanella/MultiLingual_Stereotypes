def find_stats(pre_train_df, fine_tune_df, n = 5):
    
    emotions = list(pre_train_df)
    emotions.remove("group")
    difference = pre_train_df.loc[:, pre_train_df.columns != 'group'] - fine_tune_df.loc[:, fine_tune_df.columns != 'group']
    difference["group"] = pre_train_df["group"]
    dic = {}
    
    for emotion in emotions:
        max_change = difference[emotion].nlargest(n)
        min_change = difference[emotion].nsmallest(n)
        
        max_groups = list(difference["group"].loc[max_change.index])
        min_groups = list(difference["group"].loc[min_change.index])
        
        max_min = {"max_change":max_groups,"max_change_values":list(max_change) ,"min_change":min_groups, "min_change_values":list(min_change)}
        dic[emotion] = max_min
        
    return dic