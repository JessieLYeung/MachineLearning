# Anime Recommendation Streamlit App

This small Streamlit app provides simple anime recommendations from a CSV dataset (`anime.csv`). It uses a heuristic scoring function (shared genres, type match, episode/rating differences) to return the top-N similar anime for a given title.

## What it contains
- `rec.py` — the Streamlit app and recommendation function
- `anime.csv` — the dataset (not included here if you choose to gitignore it)

## Prerequisites
- Python 3.8+ installed
- Git (optional, for repository tasks)

## Recommended Python dependencies
Create a `requirements.txt` (example below) and install them in a virtual environment.

requirements.txt example:
```
pandas
numpy
scikit-learn
streamlit
```

To create and install (PowerShell):
```powershell
# from project root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

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
- `anime.csv` includes fields used by `rec.py` (name, genre, type, episodes, rating). `rec.py` drops rows with missing numeric values (episodes, rating) or missing genre/type — so some rows may not appear in recommendations.
- If `anime.csv` is large and you don't want it in the repo, add it to `.gitignore` or use Git LFS. If you do want it in the repo, keep it in the project root (the code reads `anime.csv` by relative path).

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

## Suggested small improvements
- Add a `requirements.txt` and `.gitignore` before pushing.
- Consider preserving missing `episodes` rows (impute or remove `episodes` from required drops) so popular titles with `Unknown` episodes are included.
- Use precomputed feature vectors and cosine similarity for vector-based recommendations instead of the current heuristic scoring.

## Contact / next steps
If you'd like, I can:
- Create `requirements.txt` and `.gitignore` in this repo and commit them.
- Add a simple unit test for `recommendation_system`.
- Add a Git remote and push to your GitHub repo (paste the repo URL here and confirm).

Happy to help with any of the above — tell me which next step you want.
