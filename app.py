import os
import streamlit as st
import numpy as np

# Suppress TensorFlow logging warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

from predictor import (
    load_artifacts,
    load_prediction_model,
    predict_top_k,
    generate_text,
)

# ---------------------------------------------------------
# Page Setup & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Next Word Predictor & Text Completion",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics and clean cards
st.markdown(
    """
    <style>
    /* Main container tweaks */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    /* Hero header */
    .hero-container {
        padding: 1.5rem 1.8rem;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.08) 0%, rgba(124, 58, 237, 0.12) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        margin-bottom: 1.8rem;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0.3rem;
        background: linear-gradient(90deg, #4f46e5, #7c3aed, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 0.5rem;
    }

    /* Prediction cards */
    .prediction-box {
        background-color: rgba(99, 102, 241, 0.06);
        border: 1.5px solid rgba(99, 102, 241, 0.35);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }
    .prediction-primary-word {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4338ca;
        padding: 2px 8px;
    }
    .prediction-meta {
        font-size: 0.9rem;
        color: #6b7280;
    }

    /* Generated text highlight */
    .generated-card {
        padding: 1.3rem;
        border-radius: 12px;
        background-color: rgba(16, 185, 129, 0.06);
        border: 1px solid rgba(16, 185, 129, 0.3);
        font-size: 1.15rem;
        line-height: 1.7;
        margin-top: 1rem;
    }
    .prompt-text {
        color: #1f2937;
        font-weight: 500;
    }
    .new-tokens {
        color: #047857;
        font-weight: 700;
        background: rgba(16, 185, 129, 0.15);
        padding: 2px 6px;
        border-radius: 6px;
    }

    /* Candidate bar styling */
    .candidate-pill {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.88rem;
        background: #e0e7ff;
        color: #3730a3;
    }

    /* Sidebar info cards */
    .sidebar-info-card {
        background-color: rgba(243, 244, 246, 0.6);
        border-radius: 10px;
        padding: 0.9rem 1rem;
        border: 1px solid rgba(229, 231, 235, 1);
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Resource Caching
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading NLP artifacts & vocabulary...")
def get_nlp_artifacts():
    return load_artifacts()


@st.cache_resource(show_spinner="Loading deep learning model...")
def get_model(model_name):
    return load_prediction_model(model_name)


# Load artifacts
try:
    tokenizer, max_len, index_to_word = get_nlp_artifacts()
    vocab_size = len(tokenizer.word_index)
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "user_prompt" not in st.session_state:
    st.session_state["user_prompt"] = "The quick brown fox"

if "trigger_predict" not in st.session_state:
    st.session_state["trigger_predict"] = False


# Helper callback for example prompts
def set_prompt(text):
    st.session_state["user_prompt"] = text
    st.session_state["trigger_predict"] = True


# Helper callback for appending candidate word
def append_word(word):
    current = st.session_state["user_prompt"].rstrip()
    st.session_state["user_prompt"] = f"{current} {word}".strip()
    st.session_state["trigger_predict"] = True


# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Model Settings")

    model_choice = st.selectbox(
        "🧠 Neural Network Architecture",
        options=["LSTM", "Simple RNN"],
        index=0,
        help="LSTM handles longer-range context better than standard Simple RNN.",
    )

    mode = st.radio(
        "🎯 Prediction Mode",
        options=["Next Word Suggestion", "Multi-Word Text Completion"],
        index=0,
    )

    st.markdown("---")

    if mode == "Next Word Suggestion":
        top_k = st.slider("Top Candidate Suggestions", min_value=3, max_value=10, value=5, step=1)
        num_gen_words = 1
        temperature = 0.0
    else:
        top_k = 5
        num_gen_words = st.slider("Words to Generate", min_value=1, max_value=25, value=8, step=1)
        sampling_mode = st.selectbox(
            "Generation Strategy",
            ["Greedy (Most Likely / Deterministic)", "Temperature Sampling (Varied / Creative)"],
            index=0,
        )
        if "Temperature" in sampling_mode:
            temperature = st.slider("Temperature", min_value=0.2, max_value=1.5, value=0.7, step=0.1,
                                    help="Higher temperature yields more diverse output; lower is more deterministic.")
        else:
            temperature = 0.0

    st.markdown("---")
    st.subheader("📊 Dataset & Model Info")
    st.markdown(
        f"""
        <div class="sidebar-info-card">
            <div style="font-size: 0.85rem; color: #4b5563;"><b>Vocabulary Size:</b> {vocab_size:,} tokens</div>
            <div style="font-size: 0.85rem; color: #4b5563;"><b>Max Context Length:</b> {max_len} tokens</div>
            <div style="font-size: 0.85rem; color: #4b5563;"><b>Active Model:</b> {model_choice}</div>
            <div style="font-size: 0.85rem; color: #4b5563;"><b>Dataset:</b> Quotes & Sayings Corpus</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Load the selected model
try:
    model = get_model(model_choice)
except Exception as e:
    st.error(f"Failed to load {model_choice} model: {e}")
    st.stop()


# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">Next Word Prediction & Text Completion</div>
        <div class="hero-subtitle">
            Experience real-time natural language prediction powered by deep recurrent networks (LSTM / Simple RNN).
            Type a prompt below, explore candidate word probabilities, or automatically complete sentences.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Quick Starter Chips
# ---------------------------------------------------------
st.markdown("**💡 Quick Examples:** Click any phrase to test:")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    if st.button("🦊 The quick brown fox", use_container_width=True):
        set_prompt("The quick brown fox")
with col2:
    if st.button("❓ What is", use_container_width=True):
        set_prompt("what is")
with col3:
    if st.button("🎭 To be or not to", use_container_width=True):
        set_prompt("to be or not to")
with col4:
    if st.button("🌱 Life is a", use_container_width=True):
        set_prompt("life is a")
with col5:
    if st.button("🏆 Success is not", use_container_width=True):
        set_prompt("success is not")

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# Main Input Form
# ---------------------------------------------------------
user_input = st.text_area(
    "Enter your seed phrase or sentence:",
    value=st.session_state["user_prompt"],
    key="input_box",
    height=100,
    placeholder="Type some words here (e.g. 'Knowledge is')...",
)

# Update session state with the current text area value
st.session_state["user_prompt"] = user_input

btn_col1, btn_col2, btn_spacer = st.columns([1.5, 1.2, 4])
with btn_col1:
    action_label = "🔮 Predict Next Word" if mode == "Next Word Suggestion" else f"⚡ Generate {num_gen_words} Words"
    run_button = st.button(action_label, type="primary", use_container_width=True)
with btn_col2:
    if st.button("🧹 Clear", use_container_width=True):
        st.session_state["user_prompt"] = ""
        st.session_state["trigger_predict"] = False
        st.rerun()

should_predict = run_button or st.session_state["trigger_predict"]
# Reset trigger
st.session_state["trigger_predict"] = False


# ---------------------------------------------------------
# Prediction & Display Logic
# ---------------------------------------------------------
if should_predict:
    prompt = st.session_state["user_prompt"].strip()

    if not prompt:
        st.warning("⚠️ Please enter a phrase or select one of the quick examples above.")
    else:
        # Check if words exist in vocabulary
        words = prompt.lower().split()
        known_words = [w for w in words if w in tokenizer.word_index]
        if not known_words:
            st.warning("⚠️ None of the words in your prompt were found in the training vocabulary. Prediction might be less reliable.")

        with st.spinner("Analyzing sequence and computing probabilities..."):
            if mode == "Next Word Suggestion":
                candidates = predict_top_k(
                    model=model,
                    tokenizer=tokenizer,
                    max_len=max_len,
                    index_to_word=index_to_word,
                    text=prompt,
                    top_k=top_k,
                )

                if candidates:
                    top_word = candidates[0]["word"]
                    top_prob = candidates[0]["prob"]

                    # Top Prediction Highlight Banner
                    st.markdown(
                        f"""
                        <div class="prediction-box">
                            <div class="prediction-meta">Most Likely Next Word ({model_choice})</div>
                            <div style="display: flex; align-items: baseline; gap: 15px; margin: 8px 0;">
                                <span class="prediction-primary-word">{top_word}</span>
                                <span style="font-size: 1.1rem; font-weight: 600; color: #10b981;">
                                    {top_prob * 100:.1f}% confidence
                                </span>
                            </div>
                            <div style="font-size: 1rem; color: #374151;">
                                Complete phrase: <i>"{prompt} <b>{top_word}</b>"</i>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Quick Append Primary Word Button
                    if st.button(f'➕ Append "{top_word}" to Prompt & Continue', type="secondary"):
                        append_word(top_word)
                        st.rerun()

                    # Top Candidates Distribution
                    st.markdown(f"#### 📊 Top {len(candidates)} Candidates & Probability Breakdown")
                    st.caption("Click **+ Add** on any suggestion to append it to your prompt and predict the subsequent word.")

                    for i, cand in enumerate(candidates, 1):
                        c_word = cand["word"]
                        c_prob = cand["prob"]
                        pct = c_prob * 100

                        col_rank, col_word, col_bar, col_act = st.columns([0.6, 2.0, 5.0, 1.4])

                        with col_rank:
                            st.markdown(f"**#{i}**")

                        with col_word:
                            st.markdown(f"<span class='candidate-pill'>{c_word}</span>", unsafe_allow_html=True)

                        with col_bar:
                            st.progress(min(max(c_prob, 0.0), 1.0), text=f"{pct:.2f}%")

                        with col_act:
                            if st.button(f"+ Add", key=f"add_cand_{i}_{c_word}"):
                                append_word(c_word)
                                st.rerun()
                else:
                    st.info("No prediction available for the given input.")

            else:
                # Multi-word sequence generation
                result = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    max_len=max_len,
                    index_to_word=index_to_word,
                    seed_text=prompt,
                    n_words=num_gen_words,
                    temperature=temperature,
                )

                generated_words = result["generated_words"]
                new_tokens_str = " ".join(generated_words)
                full_text = result["full_text"]

                st.markdown("#### 📝 Generated Output")

                st.markdown(
                    f"""
                    <div class="generated-card">
                        <span class="prompt-text">{prompt}</span>
                        <span class="new-tokens"> {" ".join(generated_words)}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Stats & action row
                stat_col1, stat_col2, stat_col3 = st.columns([2, 2, 4])
                with stat_col1:
                    st.metric("Words Generated", f"+{len(generated_words)}")
                with stat_col2:
                    st.metric("Total Word Count", len(full_text.split()))
                with stat_col3:
                    st.write("")
                    if st.button("📋 Adopt Full Output into Input Box"):
                        st.session_state["user_prompt"] = full_text
                        st.rerun()
