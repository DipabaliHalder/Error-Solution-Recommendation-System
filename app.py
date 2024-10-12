import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

# Set page config for a wider layout
st.set_page_config(layout="wide", page_title="Error Solution Recommender")

# Custom CSS to improve appearance
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .big-font {
        font-size:20px !important;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Download NLTK stop words
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords')

# Load the data from CSV and perform initial cleaning
@st.cache_data
def load_and_clean_data(file):
    data = pd.read_csv(file)
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data

# Preprocess the data
@st.cache_resource
def preprocess_data(data):
    download_nltk_data()
    stop_words = stopwords.words('english')
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform(data['Error'])
    return vectorizer, tfidf_matrix

# Get recommendations using cosine similarity
def get_recommendations(query, vectorizer, tfidf_matrix, data, top_n=5):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    indices = similarities.argsort()[::-1][:top_n]
    results = data.iloc[indices].copy()
    results['Similarity'] = similarities[indices]
    return results

# Main app
def main():
    st.header("Error Solution Recommendation System")
    st.markdown("---")

    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file with 'Error' and 'Solution' columns", type="csv")

    if uploaded_file is not None:
        # Load and preprocess data
        data = load_and_clean_data(uploaded_file)
        
        if data.empty:
            st.error("The dataset is empty after cleaning. Please check your data source.")
            return
        
        if 'Error' not in data.columns or 'Solution' not in data.columns:
            st.error("The uploaded file must contain 'Error' and 'Solution' columns.")
            return
        
        vectorizer, tfidf_matrix = preprocess_data(data)

        # User input
        st.markdown("### Describe Your Issue")
        user_query = st.text_area("Enter the error message or describe the problem you're experiencing:", height=100)

        if st.button("Find Solutions"):
            if user_query:
                # Get recommendations
                recommendations = get_recommendations(user_query, vectorizer, tfidf_matrix, data)

                # Display recommendations
                st.markdown("### Recommended Solutions")
                for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                    with st.expander(f"Solution {i} (Similarity: {row['Similarity']:.2f})"):
                        st.markdown(f"<p class='big-font'>{row['Solution']}</p>", unsafe_allow_html=True)
            else:
                st.warning("Please enter an issue description before submitting.")
    else:
        st.info("Please upload a CSV file to get started.")


if __name__ == "__main__":
    main()