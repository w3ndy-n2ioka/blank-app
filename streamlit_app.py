import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import streamlit as st

# Load the data
data = pd.read_csv('Mall_Customers.csv')

# App title and description
st.title('Grand Central Mall - Customer Segmentation')
st.markdown("""
Welcome to **Grand Central Mall's** customer segmentation tool! 
This app uses **KMeans Clustering** to analyze customer behavior based on **Annual Income** and **Spending Score**.
You can filter the customers by age, gender, and adjust the number of clusters to group them.
""")

# Display dataset information and description
st.subheader('Customer Data Overview')
st.write("Here is a brief snapshot of the customer data used for segmentation:")
st.dataframe(data.head())

# Descriptive statistics
st.subheader('Descriptive Statistics')
st.write("A quick summary of the data to understand customer income and spending distribution:")
st.write(data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].describe())

# Sidebar for input parameters
st.sidebar.header('Clustering Parameters')

# Number of clusters (slider)
n_clusters = st.sidebar.slider('Number of Clusters', 2, 10, 5, help="Select the number of clusters to group the customers into.")

# Filter by age (slider)
min_age, max_age = st.sidebar.slider('Select Age Range', int(data['Age'].min()), int(data['Age'].max()), (18, 70))

# Filter by gender (multiselect)
gender_filter = st.sidebar.multiselect('Filter by Gender', options=['Male', 'Female'], default=['Male', 'Female'])

# Apply filters
filtered_data = data[(data['Age'] >= min_age) & (data['Age'] <= max_age) & (data['Gender'].isin(gender_filter))]

# Select features to cluster
X = filtered_data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Apply KMeans clustering
kmeans = KMeans(n_clusters=n_clusters)
filtered_data['Cluster'] = kmeans.fit_predict(X)

# Plot the clusters
fig = px.scatter(filtered_data, 
                 x='Annual Income (k$)', 
                 y='Spending Score (1-100)', 
                 color='Cluster', 
                 title=f'Customer Segmentation ({n_clusters} Clusters) - Grand Central Mall',
                 labels={'Annual Income (k$)': 'Annual Income', 
                         'Spending Score (1-100)': 'Spending Score'},
                 template='plotly_dark')  # Optional: dark theme for plot

# Display the clustering plot
st.subheader(f'Customer Segmentation with {n_clusters} Clusters')
st.plotly_chart(fig)

# Additional insights section
st.subheader('Insights and Recommendations')
st.write("""
From the plot, we can observe how customers are grouped based on their spending patterns and income. 
This can help the management of **Grand Central Mall** to:
- **Target High-Value Customers**: Identify high-income customers with high spending scores and create special offers or promotions for them.
- **Engage Low-Spending Customers**: Analyze why some customers with higher income have low spending scores and explore strategies to increase their spending.
""")

# Footer
st.markdown("""
---
App created by [Group 6]. Data sourced from Kaggle's Mall Customer Segmentation dataset.
""")
