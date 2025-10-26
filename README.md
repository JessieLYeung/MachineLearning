# Anime Recommendation Streamlit App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nextanime.streamlit.app)

This Streamlit app provides anime recommendations from a CSV dataset (`anime.csv`) using **cosine similarity** over feature vectors (genres, type, rating, episodes).

## Features
- **Vector-based recommendations** using cosine similarity (not just heuristics)
- **Fuzzy title matching** — handles typos and partial matches
- **Performance caching** with `@st.cache_data` for fast reloads
- **Smart data handling** — imputes missing values instead of dropping rows
- **User-friendly UI** — shows "Did you mean..." suggestions when title not found
- **Comprehensive test suite** — 8 pytest tests covering edge cases

## What it contains
- `rec.py` — the Streamlit app and recommendation function
- `anime.csv` — the dataset
- `test_rec.py` — pytest test suite
- `requirements.txt` — pinned Python dependencies
- `.gitignore` — ignore patterns for Python projects

## Prerequisites
- Python 3.8+ installed
- Git (optional, for repository tasks)

## Recommended Python dependencies
Install dependencies from `requirements.txt` (versions are pinned for reproducibility):

```powershell
# from project root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Dependencies include:
- pandas (data processing)
- numpy (numerical operations)
- scikit-learn (feature scaling, cosine similarity)
- streamlit (web UI)
- pytest (testing)

## Run tests
Run the test suite to verify everything works:
```powershell
pytest test_rec.py -v
```

All 8 tests should pass, covering:
- Data loading and preprocessing
- Exact and fuzzy title matching
- Recommendation quality and sorting
- Error handling for non-existent titles

## Run locally (Streamlit)
From the project directory run:
```powershell
cd "C:\Users\karen\OneDrive\UCF\GitHub Projects\machine learning workshop"
streamlit run rec.py
```

Then open the URL Streamlit prints (typically http://localhost:8501).

## Troubleshooting: Blank Streamlit page
- Check the VS Code terminal where you started Streamlit — Streamlit prints logs and the exact local URL there.
- If the browser shows a blank page, open the browser developer console to look for JS errors.
- Stop the server and restart if needed:
```powershell
streamlit stop 8501
streamlit run rec.py
```
- If Streamlit prompts for telemetry or email on first run it may pause; to avoid interactive prompts set config or run once manually and accept.

## Notes about the dataset
- `anime.csv` includes fields: name, genre, type, episodes, rating
- Missing episodes/ratings are imputed with median values (no rows dropped)
- Genres are multi-label encoded and used in the similarity calculation
- Feature vector combines: normalized numeric features + one-hot type + multi-hot genres

## GitHub: connect & push (quick commands)
If you already have a GitHub repo URL, from the project root:
```powershell
git init
git add .
git commit -m "Initial commit: Streamlit recommendation app"
git branch -M main
# replace with your repo URL
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```
If the remote already has commits, fetch and merge first:
```powershell
git remote add origin <url>
git fetch origin
git pull origin main --allow-unrelated-histories
# then add/commit and push
```

## Deploy to Streamlit Cloud
1. Push your project to GitHub (see above).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click "New app", choose the repo and branch (e.g. `main`) and set the file path to `rec.py`.
4. Deploy. Streamlit Cloud installs packages from `requirements.txt` automatically.

## Recent improvements
- ✅ **Caching** — `@st.cache_data` speeds up app reloads significantly
- ✅ **Pinned dependencies** — version ranges ensure reproducible deployments
- ✅ **Pytest test suite** — 8 comprehensive tests validate functionality
- ✅ **Cosine similarity** — vector-based recommendations (more robust than heuristics)
- ✅ **Fuzzy matching** — handles typos and partial titles
- ✅ **Better data handling** — imputes missing values instead of dropping rows

## Suggested next improvements
- Add autocomplete/selectbox for title selection
- Display anime posters and links (via API or web scraping)
- Add filters (genre, type, min rating)
- Show explanation for each recommendation (shared features)
- Precompute and cache features to disk for faster startup
- Deploy to Streamlit Cloud or Docker container

## Contributing
Pull requests are welcome! Please:
1. Add tests for new features
2. Run `pytest test_rec.py` to ensure all tests pass
3. Update README.md if adding new functionality

## License
MIT License - feel free to use and modify for your projects.
