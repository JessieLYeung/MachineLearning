import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cs
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import difflib
import html


def load_and_process_data():
    """Load anime.csv and preprocess features for recommendation.
    
    Returns:
        tuple: (df, features) where df is the cleaned DataFrame and features is the feature matrix
    """
    df = pd.read_csv("anime.csv")

    # Decode HTML entities in anime names (e.g., &quot; -> ", &#039; -> ')
    df['name'] = df['name'].apply(lambda x: html.unescape(str(x)) if pd.notna(x) else x)
    df['genre'] = df['genre'].apply(lambda x: html.unescape(str(x)) if pd.notna(x) else x)

    # convert numeric columns
    df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

    # fill missing genre with empty string and drop rows missing critical fields
    df['genre'] = df['genre'].fillna('')
    df = df.dropna(subset=['type'])

    # For reproducibility of indices between df and features, reset index
    df = df.reset_index(drop=True)

    # Build numeric features (episodes, rating) — impute unknown episodes with the median
    number_features = ['episodes', 'rating']
    df['episodes'] = df['episodes'].fillna(df['episodes'].median())
    df['rating'] = df['rating'].fillna(df['rating'].median())

    scaler = StandardScaler()
    scaled_numbers = scaler.fit_transform(df[number_features])

    # One-hot encode 'type'
    type_encoded = pd.get_dummies(df['type']).values

    # Multi-label binarize genres
    genres = df['genre'].apply(lambda x: [g.strip() for g in x.lower().split(',')] if x else [])
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(genres)

    # final feature matrix: scaled numbers + type one-hot + genre multi-hot
    features = np.hstack([scaled_numbers, type_encoded, genres_encoded])
    
    return df, features


# Load data on module import (will be cached in Streamlit)
df, features = load_and_process_data()


def find_closest_titles(name, names_list, n=5, cutoff=0.6):
    """Return up to n close matches from names_list for the given name (case-insensitive).
    Uses difflib with a simple cutoff for fuzzy matching."""
    lower_to_orig = {nm.lower(): nm for nm in names_list}
    lower_names = list(lower_to_orig.keys())
    name_l = name.lower()
    matches = difflib.get_close_matches(name_l, lower_names, n=n, cutoff=cutoff)
    return [lower_to_orig[m] for m in matches]


def recommendation_system(name, df_local, features_matrix, top_n=5):
    """Recommend anime similar to `name` using cosine similarity over feature vectors.

    Returns a DataFrame of the top_n recommendations (name, genre, type, episodes, rating, similarity).
    If no close match is found, returns a string message.
    """
    # exact (case-insensitive) match first
    mask = df_local['name'].str.lower() == name.lower()
    if mask.any():
        idx = int(df_local.index[mask][0])
    else:
        # try fuzzy matches
        suggestions = find_closest_titles(name, df_local['name'].tolist(), n=3, cutoff=0.5)
        if not suggestions:
            return f'"{name}" was not found (no similar titles matched).'
        # pick the top suggestion
        suggested = suggestions[0]
        idx = int(df_local.index[df_local['name'] == suggested][0])

    # compute cosine similarity between target and all items
    target_vec = features_matrix[idx].reshape(1, -1)
    sims = cs(target_vec, features_matrix).flatten()
    sims[idx] = -1  # exclude itself

    top_idx = np.argsort(sims)[::-1][:top_n]

    results = df_local.loc[top_idx, ['name', 'genre', 'type', 'episodes', 'rating']].copy()
    results['similarity'] = sims[top_idx]
    results = results.reset_index(drop=True)
    return results


if __name__ == '__main__':
    import streamlit as st

    # Cache the data loading function for performance
    @st.cache_data
    def get_data():
        """Cached data loader - only runs once per Streamlit session."""
        return load_and_process_data()
    
    # Use cached data
    df_cached, features_cached = get_data()

    st.set_page_config(page_title='Anime Recommendation', layout='centered')
    st.title('🎬 Anime Recommendation System')
    st.write("Find similar anime based on genres, type, rating, and more!")

    # Sidebar for filters
    st.sidebar.header('🔍 Filters')
    
    # Type filter
    all_types = sorted(df_cached['type'].unique().tolist())
    selected_types = st.sidebar.multiselect(
        'Anime Type',
        options=all_types,
        default=all_types,
        help='Filter by anime type (TV, Movie, etc.)'
    )
    
    # Rating filter
    min_rating = st.sidebar.slider(
        'Minimum Rating',
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.5,
        help='Only show anime with rating >= this value'
    )
    
    # Genre filter (optional - include these genres)
    all_genres = sorted(set([
        genre.strip()
        for genres_str in df_cached['genre'].dropna()
        for genre in genres_str.lower().split(',')
        if genre.strip()
    ]))
    
    include_genres = st.sidebar.multiselect(
        'Must Include Genres (optional)',
        options=all_genres,
        default=[],
        help='Recommendations must have at least one of these genres'
    )

    # Main search area
    st.subheader('Select an anime')
    
    # Autocomplete selectbox for anime selection
    anime_input = st.selectbox(
        'Search for an anime:',
        options=[''] + sorted(df_cached['name'].tolist()),
        index=0,
        help='Start typing to search'
    )
    
    # Number of recommendations
    top_n = st.slider('Number of recommendations', 1, 20, 5)

    if st.button('Get Recommendations', type='primary'):
        if not anime_input:
            st.error('Please select an anime title.')
        else:
            with st.spinner('Finding similar anime...'):
                out = recommendation_system(anime_input, df_cached, features_cached, top_n=top_n * 3)  # Get more initially for filtering
                
                if isinstance(out, str):
                    # a not-found message
                    st.error(out)
                    # also show close suggestions if any
                    close = find_closest_titles(anime_input, df_cached['name'].tolist(), n=5, cutoff=0.4)
                    if close:
                        st.write('Did you mean:')
                        for c in close:
                            st.write(f'- {c}')
                elif out.empty:
                    st.error('No recommendations found :(')
                else:
                    # Apply filters
                    filtered = out.copy()
                    
                    # Filter by type
                    if selected_types:
                        filtered = filtered[filtered['type'].isin(selected_types)]
                    
                    # Filter by rating
                    filtered = filtered[filtered['rating'] >= min_rating]
                    
                    # Filter by genres (must contain at least one)
                    if include_genres:
                        def has_required_genre(genre_str):
                            anime_genres = [g.strip() for g in str(genre_str).lower().split(',')]
                            return any(g in include_genres for g in anime_genres)
                        
                        filtered = filtered[filtered['genre'].apply(has_required_genre)]
                    
                    # Limit to requested number
                    filtered = filtered.head(top_n)
                    
                    if filtered.empty:
                        st.warning('No anime match your filters. Try adjusting the filters in the sidebar.')
                    else:
                        st.success(f'Found {len(filtered)} recommendations!')
                        
                        # Display recommendations
                        st.subheader('📺 Recommended Anime:')
                        
                        for idx, row in filtered.iterrows():
                            with st.expander(f"**{row['name']}** — Similarity: {row['similarity']:.1%}", expanded=idx==filtered.index[0]):
                                col1, col2 = st.columns([1, 2])
                                
                                with col1:
                                    st.metric('Rating', f"{row['rating']:.2f}/10")
                                    st.metric('Type', row['type'])
                                    st.metric('Episodes', int(row['episodes']) if row['episodes'] == row['episodes'] else 'Unknown')
                                
                                with col2:
                                    st.write('**Genres:**')
                                    st.write(row['genre'])
                                    st.write('**Similarity Score:**')
                                    st.progress(row['similarity'])






