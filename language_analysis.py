import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances

def drop_columns_with_same_value(df):
    same_value_columns = []
    for col in df.columns:
        if df[col].nunique() == 1:
            same_value_columns.append(col)
    df = df.drop(columns=same_value_columns)
    return df, same_value_columns


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

languages = ["Spanish", "English", "French", "Serbian-Croatian", "Greek (Modern)", "Japanese", "Catalan", "Italian", "German", "Portuguese", "Russian", "Swahili", "Hindi", "Mandarin"]
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



# Apply PCA to reduce the dimensions to 10
pca = PCA(n_components=5)
pca_data = pca.fit_transform(df_cleaned.drop("Name", axis=1))

# Compute Euclidean distance matrix based on 10 PCA dimensions for all languages
distance_matrix = euclidean_distances(pca_data, pca_data)

# Create a list to store the results
results = []

# Iterate over the language pairs
for i in range(len(df_cleaned)):
    lang1 = df_cleaned["Name"].iloc[i]
    for j in range(i+1, len(df_cleaned)):
        lang2 = df_cleaned["Name"].iloc[j]
        distance = distance_matrix[i, j]
        results.append([lang1, lang2, distance])

# Apply PCA to reduce the dimensions to 10
pca = PCA(n_components=5)
pca_data = pca.fit_transform(df_cleaned.drop("Name", axis=1))

# Compute Euclidean distance matrix based on 10 PCA dimensions for all languages
distance_matrix = euclidean_distances(pca_data, pca_data)

# Create a list to store the results
results = []

# Iterate over the language pairs
for i in range(len(df_cleaned)):
    lang1 = df_cleaned["Name"].iloc[i]
    for j in range(i+1, len(df_cleaned)):
        lang2 = df_cleaned["Name"].iloc[j]
        distance = distance_matrix[i, j]
        results.append([lang1, lang2, distance])

# Create a DataFrame from the results
df_distance = pd.DataFrame(results, columns=["Name 1", "Name 2", "Distance"])
pd.set_option('display.max_rows', df_distance.shape[0]+1)
print(df_distance.sort_values(by=["Distance"]))
"""

# Define the subset of languages
subset_languages = ["Spanish", "English", "French", "Serbian-Croatian", "Greek (Modern)", "Catalan"]

# Get the indices of the subset languages
subset_indices = [languages.index(lang) for lang in subset_languages]

# Apply PCA to reduce the dimensions to 10
pca = PCA(n_components=10)
pca_data = pca.fit_transform(df_cleaned.drop("Name", axis=1))

# Get the PCA data for the subset of languages
subset_pca_data = pca_data[subset_indices]

# Compute Euclidean distance matrix based on 10 PCA dimensions for the subset of languages
distance_matrix = euclidean_distances(subset_pca_data, subset_pca_data)

# Create a list to store the results
results = []

# Iterate over the language pairs
for i in range(len(subset_languages)):
    lang1 = subset_languages[i]
    for j in range(i+1, len(subset_languages)):
        lang2 = subset_languages[j]
        distance = distance_matrix[i, j]
        results.append([lang1, lang2, distance])

# Create a DataFrame from the results
df_distance = pd.DataFrame(results, columns=["Name 1", "Name 2", "Distance"])
print(df_distance.sort_values(by=["Distance"]))

# Apply PCA to reduce the dimensions for the subset of languages
pca = PCA(n_components=2)
subset_pca_data = pca.fit_transform(df_cleaned[df_cleaned["Name"].isin(subset_languages)].drop("Name", axis=1))

# Create a scatter plot for the subset of languages
plt.figure(figsize=(8, 6))
for i, language in enumerate(subset_languages):
    plt.scatter(subset_pca_data[i, 0], subset_pca_data[i, 1], c='blue', marker='o')
    plt.annotate(language, (subset_pca_data[i, 0], subset_pca_data[i, 1]), textcoords="offset points", xytext=(0,10), ha='center')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2-Dimension PCA for subset of languages in WALS')
plt.show()