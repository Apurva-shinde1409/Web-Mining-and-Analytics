#LOADING THE DATASET
import pandas as pd
crime_data = pd.read_csv("aggregated_crime_data_2020_2024.csv")

#EXPLORE DATA STRUCTURE
print(crime_data.info())
print(crime_data.describe())
print(crime_data['Primary Type'].unique())

#CHECK FOR MISSING DATA
print(crime_data.isnull().sum())

#DATA CLEANING AND PREPROCESSING
crime_data.dropna(subset=['Latitude', 'Longitude'], inplace=True)

#CONVERT DATE FORMATS
crime_data['Date'] = pd.to_datetime(crime_data['Date'])
crime_data['Year'] = crime_data['Date'].dt.year
crime_data['Month'] = crime_data['Date'].dt.month
crime_data['DayOfWeek'] = crime_data['Date'].dt.day_name()
crime_data['Hour'] = crime_data['Date'].dt.hour

#EXPLORATORY DATA ANALYSIS
# #Visualize crime distribution
import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend
import matplotlib.pyplot as plt
#
# crime_data['Primary Type'].value_counts().plot(kind='bar')
# plt.title('Crime Type Distribution')
# plt.show()

#MAP CRIME ACROSS DISTRICTS
import pandas as pd
import geopandas as gpd

# Load the shapefile
police_districts = gpd.read_file("/Users/apurva/Downloads/PoliceDistrict")

# Display basic information
print(police_districts.head())
print(police_districts.crs)  # Check the coordinate reference system (CRS)

# Reproject to match crime data (assuming EPSG:4326 for crime data)
# police_districts = police_districts.to_crs(epsg=4326)
#
# police_districts.plot(figsize=(10, 10), edgecolor="black", color="lightblue")
# plt.title("Chicago Police Districts")
# plt.show()

import geopandas as gpd
from shapely.geometry import Point

# Convert the crime data to GeoDataFrame
crime_data['Coordinates'] = crime_data.apply(
    lambda row: Point(row['Longitude'], row['Latitude']),
    axis=1
)
crime_gdf = gpd.GeoDataFrame(crime_data, geometry='Coordinates', crs="EPSG:4326")

# Reproject the police districts GeoDataFrame if necessary
police_districts = police_districts.to_crs(crime_gdf.crs)

# Spatial join with updated 'predicate' parameter
joined_data = gpd.sjoin(crime_gdf, police_districts, how='inner', predicate='within')

# Display the result
print(joined_data.head())

print(joined_data.columns)
district_crime_count = joined_data.groupby("District").size().reset_index(name="Crime Count")

print(district_crime_count)



# district_crime_count = joined_data.groupby("district_column_name").size().reset_index(name="Crime Count")
# print(district_crime_count)

print(police_districts.columns)
print(district_crime_count.columns)

import geopandas as gpd
import pandas as pd
# import matplotlib.pyplot as plt
#
# # Check and convert data types
# police_districts["DIST_NUM"] = police_districts["DIST_NUM"].astype(str)
# district_crime_count["District"] = district_crime_count["District"].astype(str)
# #
# # # Merge the DataFrames
# police_districts = police_districts.merge(
#     district_crime_count,
#     left_on="DIST_NUM",
#     right_on="District",
#     how="left"
# )
#
# # Plot the choropleth map
# police_districts.plot(column="Crime Count", cmap="OrRd", legend=True, figsize=(10, 10))
# plt.title("Crime Density by District")
# plt.axis("off")
# plt.show()

import geopandas as gpd
# import matplotlib.pyplot as plt
#
# # Plot the choropleth map
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# police_districts.plot(
#     column="Crime Count",
#     cmap="OrRd",
#     legend=True,
#     ax=ax,
#     edgecolor="black"
# )

# Add district labels
# for idx, row in police_districts.iterrows():
#     # Use the centroid of each district polygon for the label
#     centroid = row["geometry"].centroid
#     ax.text(
#         centroid.x, centroid.y,  # X, Y coordinates for the label
#         str(row["DIST_NUM"]),  # District number as the label
#         fontsize=8,
#         ha="center",  # Horizontal alignment
#         va="center",  # Vertical alignment
#         color="black",  # Label color
#         weight="bold"  # Label font weight
#     )

# Add title
# plt.title("Crime Density by District with Labels", fontsize=14)
# plt.axis("off")
# plt.show()
#
# # Get the top 5 districts with the highest crime count
# top_districts = district_crime_count.nlargest(5, "Crime Count")
# print("Top 5 High-Crime Districts:")
# print(top_districts)


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load the Dataset
# Assuming 'crime_data.csv' contains latitude and longitude columns
# crime_data = pd.read_csv("crime_data.csv")

# Filter out rows with missing or invalid latitude/longitude
crime_data = crime_data.dropna(subset=["Latitude", "Longitude"])

# Step 2: Normalize the Data
scaler = StandardScaler()
crime_data_scaled = scaler.fit_transform(crime_data[["Latitude", "Longitude"]])

# Step 3: Determine Optimal Number of Clusters using Elbow Method
# inertia = []
# for k in range(1, 11):
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(crime_data_scaled)
#     inertia.append(kmeans.inertia_)
#
# # Plot the Elbow Method
# plt.figure(figsize=(8, 5))
# plt.plot(range(1, 11), inertia, marker='o')
# plt.title("Elbow Method for Optimal k")
# plt.xlabel("Number of Clusters (k)")
# plt.ylabel("Inertia")
# plt.grid(True)
# plt.show()

# Step 4: Apply K-Means Clustering (Choose optimal k based on Elbow Method)
# optimal_k = 5  # Example: choose 5 clusters based on the Elbow Method
# kmeans = KMeans(n_clusters=optimal_k, random_state=42)
# crime_data["Cluster"] = kmeans.fit_predict(crime_data_scaled)
#
# # Step 5: Visualize the Clusters
# plt.figure(figsize=(10, 8))
# scatter = plt.scatter(
#     crime_data["Longitude"],
#     crime_data["Latitude"],
#     c=crime_data["Cluster"],
#     cmap="viridis",
#     s=10
# )
#
# # Add a legend to interpret clusters
# legend1 = plt.legend(*scatter.legend_elements(), title="Clusters", loc="upper right", fontsize=10)
# plt.gca().add_artist(legend1)
#
# plt.title("K-Means Clustering of High-Crime Areas in Chicago", fontsize=14)
# plt.xlabel("Longitude", fontsize=12)
# plt.ylabel("Latitude", fontsize=12)
# plt.grid(True, alpha=0.5)
# plt.show()
#
# # Save the clustered data
# crime_data.to_csv("clustered_crime_data.csv", index=False)
#
# # Optional: Visualize Clusters on a Map (Folium)
# import folium
#
# # Create a map centered on Chicago
# crime_map = folium.Map(location=[41.8781, -87.6298], zoom_start=11)
#
# # Add points to the map for each cluster
# for idx, row in crime_data.iterrows():
#     cluster_color = "blue" if row["Cluster"] == -1 else "red"  # Adjust colors as needed
#     folium.CircleMarker(
#         location=[row["Latitude"], row["Longitude"]],
#         radius=2,
#         color=cluster_color,
#         fill=True,
#         fill_opacity=0.6
#     ).add_to(crime_map)
#
# # Save the map
# crime_map.save("high_crime_clusters_map.html")

#
# import folium
# import webbrowser
#
# # Step 1: Create the map
# crime_map = folium.Map(location=[41.8781, -87.6298], zoom_start=11)
#
# # Add a sample marker to the map
# folium.Marker([41.8781, -87.6298], popup="Chicago Center").add_to(crime_map)
#
# # Step 3: Open the map in the default web browser
# webbrowser.open("/Users/apurva/PycharmProjects/Web mining and Analytics/high_crime_clusters_map.html")

import pandas as pd
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


crime_data.dropna(subset=["Latitude", "Longitude"])

# Filter data within Chicago's bounds
crime_data = crime_data[
    (crime_data["Latitude"].between(41.6, 42.1)) &
    (crime_data["Longitude"].between(-88, -87.5))
]

# Optional: Sample data if it's too large
crime_data = crime_data.sample(n=10000, random_state=42)

# # Step 2: Normalize the Data
coords = crime_data[["Latitude", "Longitude"]].values
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)
#
# # Step 3: Apply K-Means Clustering
# # Choose the number of clusters (k)
# optimal_k = 5  # Adjust based on the elbow method
# kmeans = KMeans(n_clusters=optimal_k, random_state=42)
# crime_data["Cluster"] = kmeans.fit_predict(coords_scaled)
#
# # Step 4: Visualize the Clusters
# plt.figure(figsize=(10, 8))
# plt.scatter(
#     crime_data["Longitude"],
#     crime_data["Latitude"],
#     c=crime_data["Cluster"],
#     cmap="viridis",
#     s=10
# )
#
# # Add the cluster centroids
# centroids = kmeans.cluster_centers_
# centroids_unscaled = scaler.inverse_transform(centroids)  # Convert centroids back to original scale
# plt.scatter(
#     centroids_unscaled[:, 1],  # Longitude
#     centroids_unscaled[:, 0],  # Latitude
#     c="red",
#     marker="X",
#     s=200,
#     label="Centroids"
# )
#
# # Add labels and title
# plt.title("K-Means Clustering of High-Crime Areas", fontsize=14)
# plt.xlabel("Longitude", fontsize=12)
# plt.ylabel("Latitude", fontsize=12)
# plt.legend()
# plt.grid(True, alpha=0.5)
# plt.show()

import hdbscan

hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
labels = hdbscan_clusterer.fit_predict(coords_scaled)

filtered_data = coords_scaled[labels != -1]
filtered_labels = labels[labels != -1]

if len(set(filtered_labels)) > 1:
    score = silhouette_score(filtered_data, filtered_labels)
    print(f"HDBSCAN Silhouette Score: {score}")

plt.scatter(
    crime_data["Longitude"],
    crime_data["Latitude"],
    c=crime_data["Cluster"],
    cmap="plasma",
    s=10
)
plt.title("DBSCAN Clustering Results")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()




