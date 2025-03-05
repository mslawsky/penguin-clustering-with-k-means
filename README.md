# Penguin Clustering with K-Means 🐧

**Unveiling Species and Sex Patterns with Unsupervised Learning**

This project applies K-means clustering to analyze penguin data, revealing insights into species and sex-based groupings. By examining physical characteristics, we can uncover natural clusters within penguin populations, informing ecological research and conservation efforts.

![Penguin Clusters](https://pplx-res.cloudinary.com/image/upload/v1741151552/user_uploads/XwFMQhPuGZaBTJE/species-sex-clusters.jpg)

---

## Key Insights & Analysis Findings 📊

This analysis demonstrates how K-means clustering can effectively segment penguin data, revealing natural groupings based on species and sex. The following insights provide a data-driven perspective on penguin characteristics:

### Optimal Cluster Analysis

The analysis uses the Elbow Method to determine the ideal number of clusters:

![Elbow Method](https://pplx-res.cloudinary.com/image/upload/v1741151552/user_uploads/kpnqqyRSqHRXDIS/inertia-values.jpg)

This indicates that the ideal k, based on inertia changes, is 2 or 4.

The Silhouette analysis further confirms the cluster separation:

![Silhouette Score](https://pplx-res.cloudinary.com/image/upload/v1741151552/user_uploads/eEfwjTARSKOwrna/silhouette-scores.jpg)

Based on silhouette score, k = 2 is the ideal number of clusters, indicating that 2 is the ideal number of clusters. The two most differentiated species by physical features seem to be male Gentoo and female Gentoo penguins.

Based on these cluster results, the resulting observations are:

-The most differentiated species seem to be male Gentoo and female Gentoo penguins.
-The best k is either 2 or 4. 2, according to the Silhouette analysis and 4 per the Elbow Method.

### Cluster Differentiation by Species and Sex

K-means model reveals how clusters align with species and sex combinations:

![Species and Sex Clusters](https://pplx-res.cloudinary.com/image/upload/v1741151552/user_uploads/XwFMQhPuGZaBTJE/species-sex-clusters.jpg)

This shows the clusters that were derived from K-means clustering based on physical observations. The resulting clusters are:
-Clusters predominantly Gentoo, split by gender
-Clusters predominantly Adelie or Chinstrap

This shows the clusters that were derived from K-means clustering based on physical observations. Here's the distribution.
-Clusters differentiated by species
-Clusters differentiated by species and gender

![Clusters differentiated by species](https://pplx-res.cloudinary.com/image/upload/v1741151552/user_uploads/ZRqRzmUUzKMyglv/species-cluster.jpg)

### Key Findings

* The model effectively separated Gentoo penguins based on sex
* The model groups Adelie and Chinstrap by sex
* Cluster models can serve as a baseline for evaluating penguin traits
* Further refinement will likely require a model where we include location.

---

## Ecological Applications & Strategic Impact 💡

These insights provide significant value for ecological researchers and conservation organizations and can be directly applied to several key areas:

### 1. Species Identification and Monitoring

* Identifying species based on key physical features
* Establish a baseline for monitoring changing penguin physical features as a result of external factors (like global warming)
* Understanding how different cluster combinations impact conservation

### 2. Inform Conservation

Understanding the critical feature metrics allows researchers and stakeholders to better conserve penguin populations

* Strategically allocate research
* Protect based on the identified features
* Focus on different conservation objectives based on species (like location, food sources, external feature variances, etc.)

### 3. Refine Analysis with New Models

The generated K-means clusters is a foundation for future analysis to consider other features to improve future clustering.

* Compare K-means against other machine learning or statistical models
* Expand feature list to enhance clustering

---

## Methodology & Technical Approach 🛠️

### Data Preparation & Exploration
- **Dataset**: Analysis of penguin feature data
- **Exploratory Analysis**: Examination of the relationships between all the physical features and characteristics of penguins
- **Data Quality**: Verification of data completeness and data was cleaned to remove NaN values
*The location feature was excluded for the purposed of this analysis.

### Feature Engineering
- **Feature Selection**: Dropped the location column to only focus on physical features
- **Encoding**: Encoded "sex" variable into 0 and 1

### Predictive Modeling
- **Model Selection**: Evaluated K-means
* Elbow Analysis with Inertia

---

## Technical Details & Code Implementation 💻

This project demonstrates Python skills for data analysis:


### Key Libraries & Tools
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning implementation
- **matplotlib/seaborn**: Data visualization

---

## Future Work & Enhancements 🚀

1.  **Model Refinement:**
    * Incorporate Location and Time of Year
    * Perform against other models for increased confidence
    * Include more specific biological features

2.  **External Factor Integration:**

    * Incorporate outside features
    * Impact on current climate conditions
    * Analyze the impact on species traits

---

## Repository Contents 📁

1.  **Python Analysis Files**

    *   [Penguin K-Means Notebook](penguin-k-means.py) (PY)
2.  **Datasets**

    *   [Original Dataset](penguins.csv) (CSV)
3.  **Visualizations**

    *   [Cluster Results](cluster_plot.png) (PNG)
    *   [Elbow Analysis Results](elbow_method.png) (PNG)
    *   [Silhouette Score](silhouette_score.png) (PNG)

---

## Contact & Connect 📫

For inquiries about this analysis:
- [LinkedIn Profile](https://www.linkedin.com/in/melissaslawsky/)
- [Client Results](https://melissaslawsky.com/portfolio/)
- [Tableau Portfolio](https://public.tableau.com/app/profile/melissa.slawsky1925/vizzes)
- [Email](mailto:melissa@melissaslawsky.com)

---

© Melissa Slawsky 2025. All Rights Reserved.

