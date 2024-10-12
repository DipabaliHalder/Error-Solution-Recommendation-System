import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from collections import Counter

# Set page config for a wider layout
st.set_page_config(layout="wide", page_title="Error Solution Recommender")

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

# Generate dynamic stop words
def generate_stop_words(data, top_n=50, min_df=0.1, max_df=0.9):
    download_nltk_data()
    base_stop_words = set(stopwords.words('english'))
    
    # Use CountVectorizer to get word frequencies without stop words
    count_vectorizer = CountVectorizer(stop_words='english', min_df=min_df, max_df=max_df)
    count_vectorizer.fit_transform(data['Error'])
    
    # Get the most common words
    word_freq = Counter(dict(zip(count_vectorizer.get_feature_names_out(), count_vectorizer.vocabulary_.values())))
    common_words = set([word for word, _ in word_freq.most_common(top_n)])
    
    # Combine base stop words with common words
    return list(base_stop_words.union(common_words))

# Preprocess the data
@st.cache_resource
def preprocess_data(data):
    stop_words = generate_stop_words(data)
    vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 2))
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

        if st.button(" Find Solutions"):
            if user_query:
                # Get recommendations
                recommendations = get_recommendations(user_query, vectorizer, tfidf_matrix, data)

                # Display recommendations
                st.markdown("### Recommended Solutions")
                for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                    with st.expander(f"Solution {i} (Similarity: {row['Similarity']:.2f})"):
                        st.markdown(f"<p class='big-font'>{row['Solution']}</p>", unsafe_allow_html=True)
            else:
                st.warning("⚠️ Please enter an issue description before submitting.")
    else:
        st.info("Please upload a CSV file to get started.")


if __name__ == "__main__":
    main()