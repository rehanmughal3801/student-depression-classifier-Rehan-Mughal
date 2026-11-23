# Student Depression Classifier — Streamlit

This repository contains a Streamlit interface for the supplied trained model and scaler.

## Files

- `app.py` — Streamlit prediction application
- `depression_classifier_model.pkl` — trained Random Forest classifier
- `scaler.pkl` — fitted StandardScaler
- `requirements.txt` — Python dependencies
- `.gitignore` — excludes local and temporary files

## Run locally

1. Install Python 3.10 or 3.11.
2. Open a terminal in this project folder.
3. Create and activate a virtual environment (recommended).
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the app:

   ```bash
   streamlit run app.py
   ```

## Upload to GitHub

1. Sign in at https://github.com and create a **new public repository** (or private if preferred).
2. Give it a name, for example `student-depression-classifier`.
3. Upload these files to the repository root:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `depression_classifier_model.pkl`
   - `scaler.pkl`
   - `.gitignore`
4. Commit the files.

The two `.pkl` files are required at runtime. Only upload them if you are comfortable making the trained model files public. Anyone with access to a pickle file should be treated as trusted; never load pickle files from an untrusted source.

## Deploy on Streamlit Community Cloud

1. Visit https://share.streamlit.io/ and sign in with GitHub.
2. Click **Create app**.
3. Select your repository and branch (usually `main`).
4. Set the main file path to `app.py`.
5. Click **Deploy** and wait for dependency installation.

If deployment fails, open **Manage app → Logs** and check the first error.

## Important model limitations to resolve

The supplied training code trained a `RandomForestClassifier` with 12 input columns and saved the classifier and scaler. It did **not** save the `LabelEncoder` objects or the category-to-number mappings. This app therefore asks for the original numeric codes for categorical columns. Do not guess those mappings.

The training code also does not document whether target class `0` or `1` means “depression” or “no depression”. The app intentionally shows the raw predicted class rather than assigning an unsupported meaning.

For a user-friendly production app, retrain the model using a preprocessing `Pipeline` with `OneHotEncoder` (or save all fitted encoders and their mappings), and document the target labels. The model is a screening/educational tool only, not a medical diagnosis.
