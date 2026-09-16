# =============================================================================
#  Paste this as a NEW CELL at the END of fake_job_postings.ipynb and run it.
#  It reuses the variables already in memory (x_train, y_train, tfidf,
#  encoder, scaler, top_countries), so run it after the whole notebook.
#
#  It replaces the old joblib.dump cell - don't run both.
# =============================================================================

import json, datetime, os
import joblib, sklearn, scipy, numpy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report

os.makedirs('models', exist_ok=True)

# -- 1. Train the model that actually ships -----------------------------------
# Logistic Regression, not XGBoost: it scored close on CV and it lets the
# deployed function drop the xgboost dependency entirely, which is what keeps
# the Vercel bundle under the 250MB limit.
final_model = LogisticRegression(max_iter=3000, class_weight='balanced', C=10)
final_model.fit(x_train, y_train)

proba = final_model.predict_proba(x_test)[:, 1]
print('test PR-AUC:', round(average_precision_score(y_test, proba), 4))
print(classification_report(y_test, (proba >= 0.5).astype(int),
                            target_names=['real', 'fraudulent'], zero_division=0))

# -- 2. Save the four objects the API loads -----------------------------------
joblib.dump(final_model, 'models/model.pkl', compress=3)
joblib.dump(tfidf,       'models/tfidf_vectorizer.pkl', compress=3)
joblib.dump(encoder,     'models/onehot_encoder.pkl', compress=3)
joblib.dump(scaler,      'models/standard_scaler.pkl', compress=3)

# -- 3. Save the bits the notebook never persisted ----------------------------
# top_countries is the one that silently breaks inference if it's missing:
# without it there's no way to know which countries map to 'other'.
json.dump({
    'model_type': type(final_model).__name__,
    'top_countries': list(top_countries),
    'numeric_columns': numeric_columns,
    'categorical_columns': categorical_columns,
    'threshold': 0.5,
    'trained_at': datetime.date.today().isoformat(),
    'sklearn_version': sklearn.__version__,
    'scipy_version': scipy.__version__,
    'numpy_version': numpy.__version__,
}, open('models/metadata.json', 'w'), indent=2)

print('\nPin these exact versions in requirements.txt:')
print(f'  scikit-learn=={sklearn.__version__}')
print(f'  scipy=={scipy.__version__}')
print(f'  numpy=={numpy.__version__}')
print(f'  joblib=={joblib.__version__}')

for f in sorted(os.listdir('models')):
    print(f'  models/{f}  {os.path.getsize("models/" + f) / 1e6:.2f} MB')

# -- 4. Download ---------------------------------------------------------------
try:
    import shutil
    from google.colab import files
    shutil.make_archive('models', 'zip', 'models')
    files.download('models.zip')
except ImportError:
    pass   # not on Colab, files are already in ./models
