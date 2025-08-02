from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import numpy as np
import json
import joblib

with open('semi_automated_dataset_creation/processed_decomposed_dataset.jsonl', 'r') as f:
    dataset = [json.loads(line) for line in f]

data_pair = np.array([
    (item['query'], item['S_a'] if item['label'] <= 0 else item['S_b'])
    for item in dataset
])

#TODO: Make This run on GPU

# Step 1: Build TF-IDF for queries
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data_pair[:,0])
print("X done")
# Step 2: Compute target (strategy) embeddings ONCE
encoder = SentenceTransformer('all-mpnet-base-v2')
Y = encoder.encode(data_pair[:,1], normalize_embeddings=True)
print("Y done")
# Step 3: Train regressor
reg = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, n_jobs=-1))
reg.fit(X, Y)
print("Fitting done")
# Precompute unique strategies
unique_strategies = list(set(data_pair[:,1]))
unique_embs = encoder.encode(unique_strategies, normalize_embeddings=True)

# Inference (NO encoder here)
def predict_strategy(query: str):
    x = vectorizer.transform([query])
    pred_emb = reg.predict(x)
    scores = util.cos_sim(pred_emb, unique_embs)[0]
    best_idx = scores.argmax().item()
    return unique_strategies[best_idx]

print(predict_strategy("What is the capital of France?"))  # Example query
print(predict_strategy("Explain the theory of relativity."))  # Another example query

# save model
joblib.dump(reg, 'strategy_predictor_model.pkl')
joblib.dump(vectorizer, 'strategy_vectorizer.pkl')