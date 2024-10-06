import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import streamlit as st

# Load the data
data = pd.read_csv('Mall_Customers.csv')

# Sidebar for input parameters
st.sidebar.header('KMeans Clustering Parameters')
n_clusters = st.sidebar.slider('Number of Clusters', 2, 10, 5)

# Select features to cluster
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Apply KMeans clustering
kmeans = KMeans(n_clusters=n_clusters)
data['Cluster'] = kmeans.fit_predict(X)
# Plot the clusters
fig = px.scatter(data, 
                 x='Annual Income (k$)', 
                 y='Spending Score (1-100)', 
                 color='Cluster', 
                 title='Customer Segmentation using KMeans',
                 labels={'Annual Income (k$)': 'Annual Income', 
                         'Spending Score (1-100)': 'Spending Score'})
st.plotly_chart(fig)

# Title and description
st.title('Mall Customer Segmentation')
st.write('This app uses KMeans clustering to segregate mall customers based on their spending habits and income levels.')


