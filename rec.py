import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cs
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

df = pd.read_csv("anime.csv")

df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

number_features = ['episodes', 'rating']
categorical_features = ['genre', 'type']

df = df.dropna(subset=number_features)
df = df.dropna(subset=categorical_features)

scaler = StandardScaler()
scaled_numbers = scaler.fit_transform(df[number_features])

type_encoded = pd.get_dummies(df['type'])

df['genre'] = df['genre'].fillna('')

genres = df['genre'].apply(lambda x: [g.strip() for g in x.lower().split(',')] if x else [])

# 'action', 'drama' --> ['action', 'drama']

mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(genres)

features = np.hstack([scaled_numbers, type_encoded.values, genres_encoded])

def recommendation_system(name, df, top_n = 5):
    idx_list = df.index[df['name'].str.lower() == name.lower()].tolist()
    if not idx_list:
        return (f'{name} was not found.')
    
    idx = idx_list[0]

    target = df.loc[idx]

    target_genres = set(g.strip() for g in target['genre'].lower().split(',') if g.strip())
    target_type = target['type']
    target_episodes = target['episodes']
    target_rating = target['rating']

    def genre_similarity(row):
        genres = set(g.strip() for g in row['genre'].lower().split(',') if g.strip())
        return len(target_genres.intersection(genres))
    
    df['genre_sim'] = df.apply(genre_similarity, axis=1)
    df['type_sim'] = (df['type'] == target_type).astype(int)
    df['episodes_diff'] = (df['episodes'] - target_episodes).abs()
    df['rating_diff'] = (df['rating'] - target_rating).abs()

    df['score'] = (
        df['genre_sim'] * 1000 +
        df['type_sim'] * 500 -
        df['episodes_diff'] * 10 -
        df['rating_diff'] * 5
    )

    recommendations = df[df.index != idx].sort_values('score', ascending = False).head(top_n)
    return recommendations[['name', 'genre', 'type', 'episodes', 'rating']]

import streamlit as st
import base64

st.title('anime recommendation system!')
st.write('please type in an anime title and we\'ll suggest some similar ones :)')

anime = st.text_input('enter anime title:')

if st.button('get recommendations!'):
    if anime:
        results = recommendation_system(anime, df)
        if isinstance(results, str):
            st.error(results)
        elif not results.empty:
            st.subheader('recommended anime:')
            st.dataframe(results)
        else:
            st.error('no recommendations found :(')




