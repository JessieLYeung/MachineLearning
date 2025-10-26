# Anime Recommendation Streamlit App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nextanime.streamlit.app)

A live web app that provides anime recommendations using **cosine similarity** over feature vectors (genres, type, rating, episodes).

**🌐 Try it now:** [nextanime.streamlit.app](https://nextanime.streamlit.app)

## Features
- **Vector-based recommendations** using cosine similarity (not just heuristics)
- **Fuzzy title matching** — handles typos and partial matches
- **Performance caching** with `@st.cache_data` for fast reloads
- **Smart data handling** — imputes missing values instead of dropping rows
- **User-friendly UI** — shows "Did you mean..." suggestions when title not found
- **Comprehensive test suite** — 8 pytest tests covering edge cases

## How to Use
1. Visit [nextanime.streamlit.app](https://nextanime.streamlit.app)
2. Enter an anime title (exact or partial match works!)
3. Adjust the number of recommendations (1-20)
4. Click "Get recommendations"
5. Explore similar anime with similarity scores

## What it contains
- `rec.py` — the Streamlit app and recommendation function
- `anime.csv` — the dataset (12,000+ anime titles)
- `test_rec.py` — pytest test suite
- `requirements.txt` — pinned Python dependencies
- `.gitignore` — ignore patterns for Python projects
- `.streamlit/config.toml` — custom app theme

## Technology Stack
- **Python** — Core programming language
- **Streamlit** — Web framework for the interactive UI
- **pandas** — Data processing and manipulation
- **scikit-learn** — Feature scaling and cosine similarity
- **pytest** — Testing framework

## How It Works
1. **Data Loading** — Loads 12,000+ anime from `anime.csv`
2. **Feature Engineering** — Creates vectors from:
   - Normalized numeric features (episodes, rating)
   - One-hot encoded anime type (TV, Movie, OVA, etc.)
   - Multi-hot encoded genres
3. **Similarity Calculation** — Uses cosine similarity to find similar anime
4. **Fuzzy Matching** — Handles partial/misspelled titles with difflib
5. **Smart Ranking** — Sorts by similarity score (0-1 scale)

## Recent Improvements
- ✅ **Caching** — `@st.cache_data` speeds up app reloads significantly
- ✅ **Pinned dependencies** — version ranges ensure reproducible deployments
- ✅ **Pytest test suite** — 8 comprehensive tests validate functionality
- ✅ **Cosine similarity** — vector-based recommendations (more robust than heuristics)
- ✅ **Fuzzy matching** — handles typos and partial titles
- ✅ **Better data handling** — imputes missing values instead of dropping rows

## Dataset
- **Source:** `anime.csv` (12,000+ anime titles)
- **Fields:** name, genre, type, episodes, rating, members
- **Preprocessing:** 
  - Missing episodes/ratings imputed with median values
  - Genres multi-label encoded (handles multiple genres per anime)
  - Type one-hot encoded (TV, Movie, OVA, ONA, Special, Music)

## For Developers

### Run Locally
```bash
git clone https://github.com/JessieLYeung/MachineLearning.git
cd MachineLearning
pip install -r requirements.txt
streamlit run rec.py
```

### Run Tests
```bash
pytest test_rec.py -v
```

All 8 tests should pass, covering data loading, fuzzy matching, recommendations, and edge cases.

### Contributing
Pull requests are welcome! Please:
1. Add tests for new features
2. Run `pytest test_rec.py` to ensure all tests pass
3. Update README.md if adding new functionality

## Future Improvements
- Add anime poster images and links to MyAnimeList
- Implement autocomplete/selectbox for easier title selection
- Add filters (genre, type, min rating, max episodes)
- Show explanation for each recommendation (matching features)
- Add user watchlist/favorites with session state
- Integrate collaborative filtering if user rating data available

## Author
**Karen Yeung**
- GitHub: [@JessieLYeung](https://github.com/JessieLYeung)
- Repository: [MachineLearning](https://github.com/JessieLYeung/MachineLearning)

## License
MIT License - feel free to use and modify for your projects.
