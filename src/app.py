##################################
#       K-means Project          #
##################################
### Load libraries and modules ###
# Dataframes and matrices ----------------------------------------------
import pandas as pd
import numpy as np
# Machine learning -----------------------------------------------------
from sklearn.cluster import KMeans

#########################################
# Data Preprocessing and Transformation #
#########################################
# Loading the dataset
df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv')
# Create a copy of the original dataset
df = df_raw.copy()
# New dataframe with only the 'latitude', 'longitude' and 'medincome' column
df = df.loc[:,['Latitude', 'Longitude', 'MedInc']]

#####################
# K-means algorithm #
#####################
# Instantiate the kmeans algorithm
kmeans = KMeans(n_clusters=3)
# Create cluster feature and predict the cluster by fitting the 3 columns you have
df["Cluster"] = kmeans.fit_predict(df)
# 'cluster' column to 'category' type
df["Cluster"] = df["Cluster"].astype("category")
# Clauster centers
print("The cluster centers are: ",kmeans.cluster_centers_)