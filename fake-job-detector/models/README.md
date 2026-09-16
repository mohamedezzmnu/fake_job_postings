Put the four files produced by the export cell here:

  model.pkl
  tfidf_vectorizer.pkl
  onehot_encoder.pkl
  standard_scaler.pkl
  metadata.json   (optional but recommended)

They must be committed to git - Vercel builds from the repo, and the
function reads them from disk at cold start.
