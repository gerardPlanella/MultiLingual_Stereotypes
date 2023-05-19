import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances

def compute_similarity(matrix1, matrix2):
    num_equal_rows = np.sum(np.all(matrix1 == matrix2, axis=1))
    similarity = num_equal_rows / matrix1.shape[0]
    return similarity


def compare_all_matrices(matrix_dict):
    names = list(matrix_dict.keys())
    num_matrices = len(matrix_dict)
    distances = []

    for i in range(num_matrices - 1):
        for j in range(i + 1, num_matrices):
            name1 = names[i]
            name2 = names[j]
            matrix1 = matrix_dict[name1]
            matrix2 = matrix_dict[name2]
            distance = compute_similarity(matrix1, matrix2)#cosine_distances(matrix1, matrix2)[0, 0]
            distances.append([name1, name2, distance])

    df = pd.DataFrame(distances, columns=["Name1", "Name2", "Similarity"])
    return df

def drop_columns_with_same_value(df):
    same_value_columns = []
    for col in df.columns:
        if df[col].nunique() == 1:
            same_value_columns.append(col)
    df = df.drop(columns=same_value_columns)
    return df, same_value_columns

def convert_to_matrices(df):
    one_hot_dim = df.drop("Name", axis=1).to_numpy().max()
    result = {}

    for index, row in df.iterrows():
        name = row["Name"]
        values = row.drop("Name").values
        one_hot_matrix = np.zeros((len(values), one_hot_dim + 1))

        for i, value in enumerate(values):
            one_hot_matrix[i, value] = 1

        result[name] = one_hot_matrix

    return result

def convert_to_numerical(df, drop_columns=None, skip_columns:list=["Name"]):
    df_transformed = df.copy()
    if drop_columns:
        df_transformed = df_transformed.drop(columns=drop_columns)

    value_mappings = {}
    for column in df_transformed.columns:
        if column in skip_columns or pd.api.types.is_numeric_dtype(df_transformed[column]):
            continue

        unique_values = df_transformed[column].unique()
        value_to_index = {value: index for index, value in enumerate(unique_values)}
        df_transformed[column] = df_transformed[column].map(value_to_index)
        value_mappings[column] = value_to_index

    return df_transformed, value_mappings

languages = ["Spanish", "English", "French", "Serbian-Croatian", "Greek (Modern)", "Italian", "Catalan", "Russian", "Mandarin", "German", "Japanese", "Korean", "Hindi"]
subset_names = ["Spanish", "English", "French", "Serbian-Croatian", "Greek (Modern)"]
drop_columns = ["wals_code", "iso_code", "glottocode", "countrycodes", "latitude", "longitude", "macroarea"]

df = pd.read_csv("data/language.csv")
df_filtered = df[df["Name"].isin(languages)]

# Replace "nan" strings with actual NaN values
df_filtered = df_filtered.replace("nan", np.NaN)

# Drop columns with None or NaN values for any language
df_filtered = df_filtered.dropna(axis=1, how="any")
df_filtered = df_filtered.dropna(axis=1, how="all")


df_transformed, value_mappings = convert_to_numerical(df_filtered, drop_columns)
print(value_mappings['genus'])

df_cleaned, dropped_columns = drop_columns_with_same_value(df_transformed)

print("Dropped Columns due to shared value: ")
print(dropped_columns)



"""
matrices = convert_to_matrices(df_cleaned)

#Unused now
distance_df = compare_all_matrices(matrices)
#print(distance_df.sort_values(by=["Similarity"]))


distance_df.to_csv('distance_data.csv', index=True)


subset_df = distance_df[distance_df['Name1'].isin(subset_names) & distance_df['Name2'].isin(subset_names)]

print(subset_df.sort_values(by=["Similarity"]))
"""

n_languages = len(df_cleaned.index)

# Apply PCA to reduce the dimensions
pca = PCA(n_components=2)
pca_data = pca.fit_transform(df_cleaned.drop("Name", axis=1))

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c='blue', marker='o')
for i, language in enumerate(df_cleaned["Name"]):
    plt.annotate(language, (pca_data[i, 0], pca_data[i, 1]), textcoords="offset points", xytext=(0,10), ha='center')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2-Dimension PCA for language data in WALS')
plt.show()



# Compute the Euclidean distances between the subset of PCA data
distances = euclidean_distances(pca_data)

# Create a distance DataFrame
distance_df = pd.DataFrame(distances, index=df_cleaned["Name"], columns=df_cleaned["Name"])

to_remove = list(set(languages) - set(subset_names))
# Remove rows and columns from the distance DataFrame
distance_df_filtered = distance_df.drop(to_remove, axis=0).drop(to_remove, axis=1)

print(distance_df_filtered)



