from sentence_transformers import SentenceTransformer
import numpy as np
import json
import joblib
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from ConfigSchema import ConfigSchema
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K

with open('semi_automated_dataset_creation/processed_decomposed_dataset.jsonl', 'r') as f:
    dataset = [json.loads(line) for line in f]

data_pair = np.array([
    (item['query'], item['S_a'] if item['label'] <= 0 else item['S_b'])
    for item in dataset
])

config_schema = ConfigSchema()
with open("config.cfg", "r") as cfg:
    config = {}
    for line in cfg:
        if line.strip() and not line.startswith("#"):
            key, value = line.strip().split("=")
            config[key.strip()] = value.strip()
config_schema.from_dict(config)

tokenizer = AutoTokenizer.from_pretrained(config_schema.model_name)
tokenizer.pad_token = tokenizer.eos_token

# X = tokenizer.batch_encode_plus(
#     data_pair[:, 0].tolist(),
#     padding='max_length',
#     truncation=True,
#     max_length=256,
#     return_tensors='np'
# )
# X = X['input_ids'] / tokenizer.vocab_size
# np.save('X.npy', X)
X = np.load('X.npy')
print("X done")

# encoder = SentenceTransformer('all-mpnet-base-v2')
# Y = encoder.encode(data_pair[:,1], normalize_embeddings=True)
# np.save('Y.npy', Y)
Y = np.load('Y.npy')
print("Y done")

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

print(f"Train shape: X={X_train.shape}, Y={Y_train.shape}")
print(f"Test shape:  X={X_test.shape}, Y={Y_test.shape}")

def cosine_similarity_loss(y_true, y_pred):
    """
    Loss = 1 - cosine_similarity (averaged over batch)
    """
    y_true = tf.math.l2_normalize(y_true, axis=1)
    y_pred = tf.math.l2_normalize(y_pred, axis=1)
    cosine_sim = tf.reduce_sum(y_true * y_pred, axis=1)  # batch of sims
    return 1.0 - tf.reduce_mean(cosine_sim)  # final scalar loss

def hybrid_loss(y_true, y_pred):
    cosine = cosine_similarity_loss(y_true, y_pred)
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    return cosine + 0.5 * mse  # weight mse if needed

def create_mlp(input_dim, output_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(1024, activation='relu'),
        layers.Dense(1024, activation='relu'),
        layers.Dense(output_dim)  # Linear activation for regression
    ])
    model.compile(optimizer='adam', loss=hybrid_loss, metrics=['mae',cosine_similarity_loss])
    return model

model = create_mlp(X_train.shape[1], Y_train.shape[1])
model.fit(X_train, Y_train, batch_size=16, epochs=20)


print("Testing:", model.evaluate(X_test, Y_test))

unique_strategies = np.array(list(set(data_pair[:,1])))
locs = [np.where(data_pair[:,1] == strategy)[0][0] for strategy in unique_strategies]
unique_embds = Y[locs]

def predict_strategy(query: str|list[str], verbose=False)->str:
    global Y, tokenizer, model, data_pair

    x = tokenizer.batch_encode_plus(
        query,
        padding='max_length',
        truncation=True,
        max_length=256,
        return_tensors='np'
    )
    x = x['input_ids'] / tokenizer.vocab_size
    pred_emb = model.predict(x)
    input_norm = pred_emb / np.linalg.norm(pred_emb, axis=1, keepdims=True)
    database_norm = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    cos_sim = np.dot(input_norm, database_norm.T)
    nearest_indices = np.argmax(cos_sim, axis=1).astype(int)
    if verbose:
        print(f"Nearest indices: {nearest_indices}")
        print("Similarities:",[cos_sim[i,nearest_indices[i]] for i in range(len(nearest_indices)) ] )
    return data_pair[nearest_indices, 1], cos_sim

out, sim = predict_strategy(data_pair[:,0].tolist())
