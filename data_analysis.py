import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import json

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

def get_second_elements(root_folder):
    output_dict = {}

    # walk through all subdirectories and files in the directory
    for dirpath, dirnames, filenames in os.walk(root_folder):

        # create a list for current directory
        dir_list = []

        # iterate over all files
        for filename in filenames:
            # check if file is a JSON file
            if filename == 'race.json':
                # construct full file path
                file_path = os.path.join(dirpath, filename)

                # load JSON file
                with open(file_path, 'r') as file:
                    data = json.load(file)

                    # check if data is a list and has at least two elements
                    if isinstance(data, list) and len(data) >= 2:
                        # append second element to dir_list
                        dir_list.append(data[1])

        # add dir_list to output_dict with folder name as key if it's not empty
        if dir_list:
            folder_name = os.path.basename(dirpath)
            output_dict[folder_name] = dir_list

    # convert output_dict to DataFrame
    output_df = pd.DataFrame.from_dict(output_dict, orient='index')

    return output_df

def row_means(dataframe):
    # Compute mean for each row, ignore NaN values
    means = dataframe.mean(axis=1, skipna=True)

    # Convert to list
    means_list = means.tolist()

    return means_list

if __name__ == "__main__":
    print(get_second_elements('out\\greek_finetune\spearman_correlations_RSA'))
    # print(row_means(get_second_elements('out\\spanish_finetune\spearman_correlations_RSA')))