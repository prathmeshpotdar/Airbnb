import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Streamlit page setup
st.set_page_config(page_title="Airbnb Data Dashboard", layout="wide")
st.title("🏠 Airbnb Data Dashboard")
st.caption("Interactive analysis of Airbnb listings (no calendar data — availability estimated via availability_365 if present).")

# Load data
@st.cache_data
def load_data():
    listings = pd.read_csv("listings.csv")
    # Clean price column
    if 'price' in listings.columns:
        listings['price'] = pd.to_numeric(listings['price'].astype(str)
                                          .str.replace(r'\$', '', regex=True)
                                          .str.replace(',', '', regex=True), errors='coerce')
    # Add occupancy proxy
    if 'availability_365' in listings.columns:
        listings['occupancy_rate'] = 1 - (listings['availability_365'] / 365)
    else:
        listings['occupancy_rate'] = np.nan
    return listings

listings = load_data()
sns.set(style="whitegrid", palette="pastel")

# Sidebar filters
st.sidebar.header("🔍 Filters")
neighborhoods = listings['neighbourhood_cleansed'].dropna().unique() if 'neighbourhood_cleansed' in listings.columns else []
room_types = listings['room_type'].dropna().unique() if 'room_type' in listings.columns else []

selected_neighborhood = st.sidebar.multiselect("Select Neighborhood(s)", sorted(neighborhoods))
selected_room_type = st.sidebar.multiselect("Select Room Type(s)", sorted(room_types))
price_range = st.sidebar.slider("Price Range ($)", int(listings['price'].min()), int(listings['price'].max()), (int(listings['price'].min()), int(listings['price'].max())))
min_reviews = st.sidebar.slider("Minimum Number of Reviews", 0, int(listings['number_of_reviews'].max() if 'number_of_reviews' in listings.columns else 100), 0)

# Filter data
filtered = listings.copy()
if selected_neighborhood:
    filtered = filtered[filtered['neighbourhood_cleansed'].isin(selected_neighborhood)]
if selected_room_type:
    filtered = filtered[filtered['room_type'].isin(selected_room_type)]
filtered = filtered[(filtered['price'] >= price_range[0]) & (filtered['price'] <= price_range[1])]
if 'number_of_reviews' in filtered.columns:
    filtered = filtered[filtered['number_of_reviews'] >= min_reviews]

# KPI Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Average Price ($)", f"{filtered['price'].mean():.2f}")
col2.metric("Average Occupancy Rate", f"{filtered['occupancy_rate'].mean() * 100 if not filtered['occupancy_rate'].isna().all() else 0:.1f}%")
col3.metric("Total Listings", len(filtered))

# Visualization Tabs
tabs = st.tabs(["📊 Pricing & Distribution", "📍 Location Insights", "🏠 Room Type Analysis", "💬 Sentiment & ML"])

# --- Pricing & Distribution ---
with tabs[0]:
    st.subheader("Price Distribution")
    fig, ax = plt.subplots(figsize=(8,4))
    sns.histplot(filtered['price'], bins=40, kde=True, ax=ax)
    ax.set_xlabel('Price ($)')
    st.pyplot(fig)

    if 'number_of_reviews' in filtered.columns:
        st.subheader("Price vs Number of Reviews")
        fig, ax = plt.subplots(figsize=(8,4))
        sns.scatterplot(data=filtered, x='number_of_reviews', y='price', alpha=0.6, ax=ax)
        st.pyplot(fig)

# --- Location Insights ---
with tabs[1]:
    st.subheader("Top Neighborhoods by Listings")
    if 'neighbourhood_cleansed' in filtered.columns:
        neigh_count = filtered['neighbourhood_cleansed'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(8,5))
        sns.barplot(x=neigh_count.values, y=neigh_count.index, ax=ax)
        st.pyplot(fig)

# --- Room Type Analysis ---
with tabs[2]:
    if 'room_type' in filtered.columns:
        st.subheader("Room Type Popularity")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(x='room_type', data=filtered, ax=ax)
        st.pyplot(fig)

        st.subheader("Average Price by Room Type")
        avg_price_rt = filtered.groupby('room_type')['price'].mean().sort_values(ascending=False)
        st.bar_chart(avg_price_rt)

# --- Sentiment & ML ---
with tabs[3]:
    st.subheader("Optional: Sentiment Analysis (Requires reviews.csv)")
    if st.checkbox("Run sentiment analysis on reviews.csv"):
        try:
            reviews = pd.read_csv("data/reviews.csv")
            reviews['sentiment'] = reviews['comments'].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
            merged = reviews.merge(listings[['id','neighbourhood_cleansed','room_type']], left_on='listing_id', right_on='id', how='left')
            sentiment_avg = merged.groupby('room_type')['sentiment'].mean()
            st.bar_chart(sentiment_avg)
        except Exception as e:
            st.error(f"Error loading reviews.csv: {e}")

    st.divider()

    st.subheader("💡 Predict Price (Simple ML Model)")
    if st.checkbox("Train and Predict using Linear Regression"):
        numeric_cols = ['number_of_reviews', 'availability_365']
        features = [col for col in numeric_cols if col in listings.columns]
        if 'price' in listings.columns and features:
            model_df = listings[features + ['price']].dropna()
            X = model_df[features]
            y = model_df['price']
            model = LinearRegression().fit(X, y)
            st.success("Model trained successfully!")
            sample = {}
            for f in features:
                sample[f] = st.number_input(f"Enter {f}", float(X[f].min()), float(X[f].max()), float(X[f].mean()))
            sample_df = pd.DataFrame([sample])
            pred = model.predict(sample_df)[0]
            st.info(f"Predicted Price: ${pred:.2f}")

# Export data
st.download_button("📤 Export Filtered Data (CSV)", data=filtered.to_csv(index=False), file_name="filtered_airbnb_data.csv", mime="text/csv")