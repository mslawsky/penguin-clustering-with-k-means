#!/usr/bin/env python
# coding: utf-8

# # Activity: Build a K-means model 
# 
# ## **Introduction**
# 
# K-means clustering is very effective when segmenting data and attempting to find patterns. Because clustering is used in a broad array of industries, becoming proficient in this process will help you expand your skillset in a widely applicable way.   
# 
# In this activity, you are a consultant for a scientific organization that works to support and sustain penguin colonies. You are tasked with helping other staff members learn more about penguins in order to achieve this mission. 
# 
# The data for this activity is in a spreadsheet that includes datapoints across a sample size of 345 penguins, such as species, island, and sex. Your will use a K-means clustering model to group this data and identify patterns that provide important insights about penguins.
# 
# **Note:** Because this lab uses a real dataset, this notebook will first require basic EDA, data cleaning, and other manipulations to prepare the data for modeling. 

# ## **Step 1: Imports** 
# 

# Import statements including `K-means`, `silhouette_score`, and `StandardScaler`.

# In[1]:


# Import standard operational packages.

# Important tools for modeling and evaluation.

# Import visualization packages.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


# `Pandas` is used to load the penguins dataset, which is built into the `seaborn` library. The resulting `pandas` DataFrame is saved in a variable named `penguins`. As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[2]:


# RUN THIS CELL TO IMPORT YOUR DATA. 

# Save the `pandas` DataFrame in variable `penguins`. 

penguins = pd.read_csv("penguins.csv")


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the `load_dataset` function. 
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# The function is from seaborn (`sns`). It should be passed in the dataset name `'penguins'` as a string. 
# 
# </details>

# Now, review the first 10 rows of data.
# 

# In[3]:


# Review the first 10 rows.

print(penguins.head())


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the `head()` method.
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# By default, the method only returns five rows. To change this, specify how many rows `(n = )` you want.
# 
# </details>

# ## **Step 2: Data exploration** 
# 
# After loading the dataset, the next step is to prepare the data to be suitable for clustering. This includes: 
# 
# *   Exploring data
# *   Checking for missing values
# *   Encoding data 
# *   Dropping a column
# *   Scaling the features using `StandardScaler`

# ### Explore data
# 
# To cluster penguins of multiple different species, determine how many different types of penguin species are in the dataset.

# In[4]:


# Find out how many penguin types there are.

penguin_species = penguins['species'].unique()
num_species = len(penguin_species)

# Print the number and names of the species
print(f"Number of penguin types/species: {num_species}")
print(f"Penguin species: {penguin_species}")


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the `unique()` method.
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the `unique()` method on the column `'species'`.
# 
# </details>

# In[5]:


# Find the count of each species type.

species_counts = penguins['species'].value_counts()

# Print the results
print("Count of each penguin species:")
print(species_counts)


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the `value_counts()` method.
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the `value_counts()` method on the column `'species'`.
# 
# </details>

# **Question:** How many types of species are present in the dataset?

# Count of each penguin species:
# -Adelie       152
# -Gentoo       124
# -Chinstrap     68
# 

# **Question:** Why is it helpful to determine the perfect number of clusters using K-means when you already know how many penguin species the dataset contains?

# In essence, while you could simply set k=3 based on prior knowledge, finding the optimal k through data analysis represents a more rigorous approach that validates your understanding of the data and might reveal unexpected patterns.

# ### Check for missing values

# An assumption of K-means is that there are no missing values. Check for missing values in the rows of the data. 

# In[6]:


# Check for missing values.

print(penguins.isnull().sum())


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the `isnull` and `sum` methods. 
# 
# </details>

# Now, drop the rows with missing values and save the resulting pandas DataFrame in a variable named `penguins_subset`.

# In[8]:


# Drop rows with missing values
# Save DataFrame in variable `penguins_subset`
penguins_subset = penguins.dropna()

# Check if any missing values remain
print("Missing values after dropping NaN rows:")
print(penguins_subset.isnull().sum())

# View the shape of the data before and after dropping NaN values
print(f"\nOriginal data shape: {penguins.shape}")
print(f"Clean data shape: {penguins_subset.shape}")

# View first few rows of the cleaned data
print("\nFirst few rows of penguins_subset:")
print(penguins_subset.head())


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use `dropna`. Note that an axis parameter passed in to this function should be set to 0 if you want to drop rows containing missing values or 1 if you want to drop columns containing missing values. Optionally, `reset_index` may also be used to avoid a SettingWithCopy warning later in the notebook. 
# </details>

# Next, check to make sure that `penguins_subset` does not contain any missing values.

# In[9]:


# Check for missing values.


# Check for missing values
print(penguins_subset.isnull().sum())

# View the first few rows of the cleaned data
print(penguins_subset.head())


# Now, review the first 10 rows of the subset.

# In[10]:


# View first 10 rows.

print(penguins_subset.head(10))



# ### Encode data
# 
# Some versions of the penguins dataset have values encoded in the sex column as 'Male' and 'Female' instead of 'MALE' and 'FEMALE'. The code below will make sure all values are ALL CAPS. 
# 

# In[11]:


penguins_subset['sex'] = penguins_subset['sex'].str.upper()


# K-means needs numeric columns for clustering. Convert the categorical column `'sex'` into numeric. There is no need to convert the `'species'` column because it isn't being used as a feature in the clustering algorithm. 

# In[12]:


# Now map 'MALE' to 0 and 'FEMALE' to 1
sex_mapping = {'MALE': 0, 'FEMALE': 1}
penguins_subset['sex'] = penguins_subset['sex'].map(sex_mapping)

# Verify the changes
print(penguins_subset[['sex']].head())


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the `get_dummies` function. 
# 
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# The `drop_first` parameter should be set to `True`. This removes redundant data. The `columns` parameter can **optionally** be set to `['sex']` to specify that only the `'sex'` column gets this operation performed on it. 
# 
# </details>

# ### Drop a column
# 
# Drop the categorical column `island` from the dataset. While it has value, this notebook is trying to confirm if penguins of the same species exhibit different physical characteristics based on sex. This doesn't include location.
# 
# Note that the `'species'` column is not numeric. Don't drop the `'species'` column for now. It could potentially be used to help understand the clusters later. 

# In[13]:


# Drop the island column.

penguins_subset = penguins_subset.drop(columns=['island'])

# Verify if the 'island' column is dropped
print(penguins_subset.columns)


# ### Scale the features
# 
# Because K-means uses distance between observations as its measure of similarity, it's important to scale the data before modeling. Use a third-party tool, such as scikit-learn's `StandardScaler` function. `StandardScaler` scales each point xᵢ by subtracting the mean observed value for that feature and dividing by the standard deviation:
# 
# x-scaled = (xᵢ – mean(X)) / σ
# 
# This ensures that all variables have a mean of 0 and variance/standard deviation of 1. 
# 
# **Note:** Because the species column isn't a feature, it doesn't need to be scaled. 
# 
# First, copy all the features except the `'species'` column to a DataFrame `X`. 

# In[14]:


# Exclude `species` variable from X

X = penguins_subset.drop(columns=['species'])

# Verify that 'species' has been excluded
print(X.head())


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use`drop()`.
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Select all columns except `'species'.`The `axis` parameter passed in to this method should be set to `1` if you want to drop columns.
# </details>

# Scale the features in `X` using `StandardScaler`, and assign the scaled data to a new variable `X_scaled`. 

# In[18]:


#Scale the features.
#Assign the scaled data to variable `X_scaled`.

X = penguins_subset.drop(columns=['species'])  

# Drop species column for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Instantiate StandardScaler to transform the data in a single step.
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use the `.fit_transform()` method and pass in the data as an argument.
# </details>

# ## **Step 3: Data modeling** 

# Now, fit K-means and evaluate inertia for different values of k. Because you may not know how many clusters exist in the data, start by fitting K-means and examining the inertia values for different values of k. To do this, write a function called `kmeans_inertia` that takes in `num_clusters` and `x_vals` (`X_scaled`) and returns a list of each k-value's inertia.
# 
# When using K-means inside the function, set the `random_state` to `42`. This way, others can reproduce your results.

# In[20]:


# Fit K-means and evaluate inertia for different values of k.

# Function to clean data
def clean_data(X_vals):
    # Remove rows where any NaN or infinite value exists
    X_cleaned = X_vals[~np.isnan(X_vals).any(axis=1)]
    X_cleaned = X_cleaned[~np.isinf(X_cleaned).any(axis=1)]
    
    # Check the shape of cleaned data
    print(f"Cleaned data shape: {X_cleaned.shape}")
    
    return X_cleaned


# Use the `kmeans_inertia` function to return a list of inertia for k=2 to 10.

# In[21]:


# Return a list of inertia for k=2 to 10.

from sklearn.cluster import KMeans

def kmeans_inertia(num_clusters, x_vals):
    inertia = []
    for k in num_clusters:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(x_vals)
        inertia.append(kmeans.inertia_)
    return inertia

k_values = range(2, 11)
inertia_values = kmeans_inertia(k_values, X_scaled)

# Print the inertia values
print("Inertia values for k=2 to 10:", inertia_values)


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Review the material about the `kmeans_inertia` function. 
# </details>

# Next, create a line plot that shows the relationship between `num_clusters` and `inertia`.
# Use either seaborn or matplotlib to visualize this relationship. 

# In[22]:


# Create a line plot.

plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), inertia_values, marker='o', color='b', linestyle='--')
plt.title('Inertia for Different Values of k', fontsize=14)
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Inertia', fontsize=12)
plt.xticks(range(2, 11))
plt.grid(True)
plt.show()


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use `sns.lineplot`.
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Include `x=num_clusters` and `y=inertia`.
# </details>

# **Question:** Where is the elbow in the plot?

# The elbow in the plot occurs at k=4.

# ## **Step 4: Results and evaluation** 

# Now, evaluate the silhouette score using the `silhouette_score()` function. Silhouette scores are used to study the distance between clusters. 

# Then, compare the silhouette score of each value of k, from 2 through 10. To do this, write a function called `kmeans_sil` that takes in `num_clusters` and `x_vals` (`X_scaled`) and returns a list of each k-value's silhouette score.

# In[23]:


# Evaluate silhouette score.
# Write a function to return a list of each k-value's score.

from sklearn.metrics import silhouette_score

def kmeans_sil(num_clusters, x_vals):
    sil_scores = []
    for k in num_clusters:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(x_vals)
        labels = kmeans.labels_
        sil_score = silhouette_score(x_vals, labels)
        sil_scores.append(sil_score)
    return sil_scores

# Calculate silhouette scores
sil_scores = kmeans_sil(k_values, X_scaled)

# Print the silhouette scores
print("Silhouette Scores for k=2 to 10:", sil_scores)


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Review the `kmeans_sil` function video.
# </details>

# Next, create a line plot that shows the relationship between `num_clusters` and `sil_score`.
# Use either seaborn or matplotlib to visualize this relationship. 

# In[24]:


# Create a line plot.

plt.figure(figsize=(8, 6))
plt.plot(k_values, sil_scores, marker='o', color='r', linestyle='--')
plt.title('Silhouette Scores for Different Values of k', fontsize=14)
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.xticks(k_values)
plt.grid(True)
plt.show()


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use `sns.lineplot`.
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Include `x=num_clusters` and `y=sil_score`.
# </details>

# **Question:** What does the graph show?

# The graph shows the silhouette scores for different values of kk (number of clusters) in a K-means model. Here's the breakdown:
# Key Observations:
# 
#     Highest Silhouette Score:
# 
#         The peak score is 0.525 at k=2k=2, indicating the best-defined clusters when using 2 clusters.
# 
#         Silhouette scores decrease steadily as kk increases beyond 2.
# 
#     Interpretation:
# 
#         k=2k=2 is optimal according to the silhouette score metric, as it has the highest value.
# 
#         Scores decline for k=3k=3 to k=10k=10, suggesting that increasing the number of clusters leads to poorer separation between groups.
# 
#     Contradiction with Inertia:
# 
#         Earlier inertia analysis suggested an elbow at k=4k=4, but silhouette scores favor k=2k=2.
# 
#         This discrepancy highlights that different metrics prioritize different aspects of clustering:
# 
#             Inertia: Focuses on minimizing within-cluster variance (compactness).
# 
#             Silhouette Score: Balances cluster cohesion and separation.
# 
#     Practical Implications:
# 
#         If the goal is to maximize cluster separation/cohesion, k=2k=2 is best.
# 
#         If domain knowledge (e.g., 3 penguin species) suggests k=3k=3, the model’s lower silhouette score at k=3k=3 implies the features used may not align perfectly with biological distinctions.
# 
# Graph Summary:
# 
#     X-axis: Number of clusters (kk) from 2 to 10.
# 
#     Y-axis: Silhouette scores (0.375 to 0.525).
# 
#     Takeaway: The data naturally forms two well-separated groups, but additional clusters (e.g., k=3k=3) may not meaningfully improve the model’s ability to distinguish penguin subgroups based on the current features.

# ### Optimal k-value

# To decide on an optimal k-value, fit a six-cluster model to the dataset. 

# In[25]:


# Fit a 6-cluster model.

optimal_k = 4  # Adjust based on your analysis
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X_scaled)

# Add cluster labels to the DataFrame
penguins_subset['cluster'] = kmeans.labels_


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Make an instance of the model with `num_clusters = 6` and use the `fit` function on `X_scaled`. 
# </details>
# 
# 
# 

# Print out the unique labels of the fit model.

# In[26]:


# Print unique labels.

print(penguins_subset.groupby(['cluster', 'species']).size())

# Group by cluster, species, and sex
print(penguins_subset.groupby(['cluster', 'species', 'sex']).size())


# Now, create a new column `cluster` that indicates cluster assignment in the DataFrame `penguins_subset`. It's important to understand the meaning of each cluster's labels, then decide whether the clustering makes sense. 
# 
# **Note:** This task is done using `penguins_subset` because it is often easier to interpret unscaled data.

# In[27]:


# Create a new column `cluster`.

print(penguins_subset.groupby(['cluster', 'species']).size())


# Use `groupby` to verify if any `'cluster'` can be differentiated by `'species'`.

# In[28]:


# Verify if any `cluster` can be differentiated by `species`.

print(penguins_subset.groupby(['cluster', 'species', 'sex']).size())


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use `groupby(by=['cluster', 'species'])`. 
# 
# </details>
# 

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# 
# Use an aggregation function such as `size`.
# 
# </details>

# Next, interpret the groupby outputs. Although the results of the groupby show that each `'cluster'` can be differentiated by `'species'`, it is useful to visualize these results. The graph shows that each `'cluster'` can be differentiated by `'species'`. 
# 
# **Note:** The code for the graph below is outside the scope of this lab. 

# In[29]:


penguins_subset.groupby(by=['cluster', 'species']).size().plot.bar(title='Clusters differentiated by species',
                                                                   figsize=(6, 5),
                                                                   ylabel='Size',
                                                                   xlabel='(Cluster, Species)');


# Use `groupby` to verify if each `'cluster'` can be differentiated by `'species'` AND `'sex_MALE'`.

# In[31]:


# Verify if each `cluster` can be differentiated by `species' AND `sex_MALE`.

# Group by cluster, species, and sex to see counts
cluster_species_sex = penguins_subset.groupby(['cluster', 'species', 'sex']).size().reset_index(name='count')

# Pivot the table for better readability
pivot_table = cluster_species_sex.pivot_table(
    index=['cluster', 'species'],
    columns='sex',
    values='count',
    fill_value=0
).rename(columns={0: 'Male', 1: 'Female'})

print(pivot_table)


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use `groupby(by=['cluster','species', 'sex_MALE'])`. 
# </details>
# 

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use an aggregation function such as `size`.
# </details>

# **Question:** Are the clusters differentiated by `'species'` and `'sex_MALE'`?

# Yes, the clusters are differentiated by both species and sex (male/female).

# Finally, interpret the groupby outputs and visualize these results. The graph shows that each `'cluster'` can be differentiated by `'species'` and `'sex_MALE'`. Furthermore, each cluster is mostly comprised of one sex and one species. 
# 
# **Note:** The code for the graph below is outside the scope of this lab. 

# In[33]:


penguins_subset.groupby(by=['cluster', 'species', 'sex']).size().unstack(level='species', fill_value=0).plot.bar(
    title='Clusters differentiated by species and sex',
    figsize=(6, 5),
    ylabel='Size',
    xlabel='(Cluster, Sex)'
)

plt.legend(bbox_to_anchor=(1.3, 1.0))
plt.show()


# ## **Considerations**
# 
# 
# **What are some key takeaways that you learned during this lab? Consider the process you used, key tools, and the results of your investigation.**
# Key Takeaways:
# 
# 1.Data Preprocessing is Crucial: Handling missing values, encoding categorical variables (e.g., sex), and scaling features (via StandardScaler) were essential steps to prepare the data for clustering.
# 
# 2.Choosing Optimal kk is Context-Dependent:
# 
# -Inertia suggested k=4k=4 (elbow method), while silhouette scores favored k=2k=2. This highlights the need to balance mathematical metrics with domain knowledge.
# 
# -For penguin data, k=4k=4 aligned better with biological distinctions (species and sex).
# 
# 3.Clusters Reflect Biological Realities:
# 
# -Gentoo penguins formed distinct male/female clusters.
# 
# -Adelie and Chinstrap penguins were grouped by sex but merged across species, likely due to overlapping physical traits.
# 
# 4.Tool Proficiency: Mastery of pandas (data cleaning), scikit-learn (K-means, scaling), and visualization libraries (matplotlib, seaborn) streamlined the analysis.
# 
# 
# **What summary would you provide to stakeholders?**
# 
# Stakeholder Summary:
# Our analysis successfully grouped penguins into clusters that align with biological categories:
# 
# -Gentoo Penguins: Separated into distinct male and female clusters.
# 
# -Adelie/Chinstrap Penguins: Grouped by sex, though not split by species, suggesting shared physical traits.
# 
# Implications:
# 
# -These clusters can guide targeted conservation strategies (e.g., sex-specific interventions for Gentoos).
# 
# -Further investigation into Adelie/Chinstrap overlap may refine species-specific management.
# 
# Next Steps:
# 
# -Validate clusters with additional data (e.g., genetic or behavioral metrics).
# 
# -Explore alternative models (e.g., hierarchical clustering) to better distinguish Adelie/Chinstrap.
# 
# This work underscores the value of clustering in uncovering actionable insights for penguin conservation.
# 
# 
# 
# 

# ### References
# 
# [Gorman, Kristen B., et al. “Ecological Sexual Dimorphism and Environmental Variability within a Community of Antarctic Penguins (Genus Pygoscelis).” PLOS ONE, vol. 9, no. 3, Mar. 2014, p. e90081. PLoS Journals](https://doi.org/10.1371/journal.pone.0090081)
# 
# [Sklearn Preprocessing StandardScaler scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged
