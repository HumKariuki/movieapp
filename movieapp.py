import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
movie_data = pd.read_csv("movies.csv")

# Selecting the best features from the dataset
selected_features = ['genres', 'keywords', 'original_language', 'popularity', 'release_date', 
                     'revenue', 'runtime', 'cast', 'director', 'vote_average', 'vote_count']

# Replace the null values with an empty string
for feature in selected_features:
    movie_data[feature] = movie_data[feature].fillna('')

# Combine all the selected columns into a single feature
combined_features = movie_data[selected_features].apply(lambda x: ' '.join(map(str, x)), axis=1)

# Convert text data to feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Calculate cosine similarity scores
similarity = cosine_similarity(feature_vectors)

# Streamlit app starts here
st.title('Movie Recommendation System')

# Search bar for movie name input
movie_name = st.text_input('Enter your Favorite Movie Name:')

# Button to trigger movie search
if st.button('Search'):
    # Get a list of all movie titles
    list_for_all_titles = movie_data['title'].tolist()

    # Find the closest match to the user's input
    find_close_match = difflib.get_close_matches(movie_name, list_for_all_titles)

    # Check if there's a close match
    if find_close_match:
        # Get the closest match
        close_match = find_close_match[0]

        # Find the index of the closest match in the dataset
        index_of_the_movie = movie_data[movie_data.title == close_match]['index'].values[0]

        # Calculate similarity scores
        similarity_score = list(enumerate(similarity[index_of_the_movie]))

        # Sort similar movies based on similarity scores
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        # Display the recommended movies
        st.write('Movies Suggested For You:\n')
        i = 1
        for movie in sorted_similar_movies:
            index = movie[0]
            title_from_index = movie_data[movie_data.index == index]['title'].values[0]
            if i < 31:
                st.write(f"{i}. {title_from_index}")
                i += 1
    else:
        st.write("Sorry, we couldn't find any movie similar to your input.")
