import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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

if __name__ == "__main__":
    racial_minorities =  ["Indians", "Russians", "Africans", "Black people",
        "Mexicans", "Blacks","Arabs",
        "Black Americans", "Black men",
        "Asian women", "Asian men", "Black women", 
        "Indian men", "Indian women", "Chinese men", "Chinese women",
        "Indian parents", "Asian parents", "Black fathers", "Latinas", "Latinos",
        "Asian kids", "Black kids", "African Americans",
        "Nigerians", "Ethiopians","Sudanese people", "Afghans", "Iraqis",
        "Somalis", "Iranian people", "Iranians",
        "Ghanaians",
        "Syrians", "Pakistanis",
        "Romanians", "Ecuadorians",
        "Turkish people"
        ]
    
    df_1 = pd.read_csv('out\pretrained_roberta\emotion_profiles\\english\\race.csv', index_col = False)
    df_1.rename( columns={'Unnamed: 0':'group'}, inplace=True )
    print(df_1)
    df = df_1[df_1.index.isin(racial_minorities)]
    print(df)
    # df_2 = pd.read_csv('out\\finetuned_stereoset_english\emotion_profiles\\French\\race.csv', index_col = False)
    # df_2.rename( columns={'Unnamed: 0':'group'}, inplace=True )

    # print(find_stats(df_1, df_2))

    # df_1 = pd.read_csv('out\\finetuned_stereoset_english\emotion_profiles\English\\race.csv', index_col = False)
    # df_1.rename(columns={'Unnamed: 0':'group'}, inplace=True )
    # df_1.set_index('group', inplace=True)

    # print(df_1.head())
    # plt.figure(figsize=(10, 10))
    # sns.heatmap(df_1, cmap='viridis')
    # plt.show()