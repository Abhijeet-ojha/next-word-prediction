import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")
MAX_LEN_PATH = os.path.join(BASE_DIR, "max_len.pkl")
LSTM_MODEL_PATH = os.path.join(BASE_DIR, "lstm_model.h5")
RNN_MODEL_PATH = os.path.join(BASE_DIR, "rnn_model.h5")


def load_artifacts():
    """Load tokenizer, max_len, and build reverse index."""
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer file not found at {TOKENIZER_PATH}")
    if not os.path.exists(MAX_LEN_PATH):
        raise FileNotFoundError(f"Max len file not found at {MAX_LEN_PATH}")

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    with open(MAX_LEN_PATH, "rb") as f:
        max_len = pickle.load(f)

    index_to_word = {index: word for word, index in tokenizer.word_index.items()}
    return tokenizer, max_len, index_to_word


def load_prediction_model(model_type="LSTM"):
    """Load selected Keras model."""
    if model_type.upper() == "LSTM":
        path = LSTM_MODEL_PATH
    else:
        path = RNN_MODEL_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")

    model = load_model(path, compile=False)
    return model


def predict_top_k(model, tokenizer, max_len, index_to_word, text, top_k=5):
    """
    Predict top K most probable next words given input text.
    Returns list of dicts with word, probability (0-1), and token index.
    """
    cleaned_text = text.strip().lower()
    if not cleaned_text:
        return []

    seq = tokenizer.texts_to_sequences([cleaned_text])[0]
    if not seq:
        return []

    # Pad sequence to match training configuration (pre-padding)
    seq_padded = pad_sequences([seq], maxlen=max_len, padding="pre")
    predictions = model.predict(seq_padded, verbose=0)[0]

    # Get top K indices sorted descending by probability
    top_indices = np.argsort(predictions)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        word = index_to_word.get(int(idx), None)
        if word:
            results.append({
                "word": word,
                "prob": float(predictions[idx]),
                "index": int(idx)
            })

    return results


def generate_text(model, tokenizer, max_len, index_to_word, seed_text, n_words=5, temperature=1.0):
    """
    Generate next n_words starting from seed_text.
    Supports greedy (temperature=0 or None) or temperature-based sampling.
    """
    current_text = seed_text.strip()
    generated_tokens = []

    for _ in range(n_words):
        # Use last max_len-1 tokens
        tokens = current_text.lower().split()
        input_slice = " ".join(tokens[-(max_len - 1):])

        seq = tokenizer.texts_to_sequences([input_slice])[0]
        if not seq:
            break

        seq_padded = pad_sequences([seq], maxlen=max_len, padding="pre")
        preds = model.predict(seq_padded, verbose=0)[0]

        if temperature <= 0.05:
            # Deterministic greedy choice
            predicted_index = int(np.argmax(preds))
        else:
            # Temperature sampling
            preds = np.asarray(preds, dtype=np.float64)
            preds = np.log(preds + 1e-10) / temperature
            exp_preds = np.exp(preds - np.max(preds))
            preds = exp_preds / np.sum(exp_preds)
            predicted_index = int(np.random.choice(len(preds), p=preds))

        next_word = index_to_word.get(predicted_index, "")
        if not next_word or next_word.strip() == "":
            break

        generated_tokens.append(next_word)
        current_text += " " + next_word

    return {
        "full_text": current_text,
        "generated_words": generated_tokens,
        "count": len(generated_tokens)
    }
