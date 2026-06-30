"""
AI Sentiment & Emotion Analyzer
--------------------------------
A Streamlit web app that lets users analyze sentiment (positive / negative / neutral)
and detect emotions (joy, anger, fear, sadness, trust, etc.) from:
  - A pasted URL (news article / blog / Amazon review page)
  - Pasted raw text
  - An uploaded CSV/TXT file containing multiple texts (e.g. reviews, tweets)

Sentiment engine: VADER (lexicon, great for social media) + a pretrained
HuggingFace transformer (DistilBERT fine-tuned on SST-2) for higher accuracy.
Emotion engine: NRCLex (NRC Emotion Lexicon) for fine-grained emotions.

Run with:  streamlit run app.py
"""

import re
import io
import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime

import trafilatura
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex

# Transformer model is loaded lazily (it's heavier) and cached
@st.cache_resource(show_spinner="Loading transformer model (first run only)...")
def load_transformer():
    from transformers import pipeline
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

vader = SentimentIntensityAnalyzer()

# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def extract_text_from_url(url: str) -> str:
    """Pull the main readable text out of a news/article/review page."""
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        # fallback: plain requests + trafilatura extraction on raw HTML
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        downloaded = resp.text
    text = trafilatura.extract(downloaded)
    return text or ""


def classify_sentiment(text: str, transformer_clf) -> dict:
    """Combine VADER (handles neutrality + negation/slang well) with a
    transformer model (higher accuracy on longer, well-formed text)."""
    text = text.strip()
    if not text:
        return {"label": "neutral", "vader_compound": 0.0, "transformer_label": None, "transformer_score": None}

    vader_scores = vader.polarity_scores(text)
    compound = vader_scores["compound"]

    # VADER-based label with a neutral band
    if compound >= 0.05:
        vader_label = "positive"
    elif compound <= -0.05:
        vader_label = "negative"
    else:
        vader_label = "neutral"

    # Transformer (binary pos/neg) — only run on a reasonably sized chunk
    truncated = text[:1500]
    try:
        result = transformer_clf(truncated)[0]
        transformer_label = result["label"].lower()
        transformer_score = round(result["score"], 3)
    except Exception:
        transformer_label, transformer_score = None, None

    # Final decision: if VADER says neutral, trust it (transformer can't say neutral).
    # Otherwise, use transformer label if it's confident (>0.75), else fall back to VADER.
    if vader_label == "neutral":
        final_label = "neutral"
    elif transformer_label and transformer_score and transformer_score > 0.75:
        final_label = "positive" if transformer_label == "positive" else "negative"
    else:
        final_label = vader_label

    return {
        "label": final_label,
        "vader_compound": round(compound, 3),
        "transformer_label": transformer_label,
        "transformer_score": transformer_score,
    }


def detect_emotions(text: str) -> dict:
    """Return normalized emotion frequencies using the NRC Emotion Lexicon."""
    text = text.strip()
    if not text:
        return {}
    obj = NRCLex(text)
    freqs = obj.affect_frequencies
    # Drop the 'positive'/'negative' meta-keys NRCLex includes; keep core emotions
    core = {k: v for k, v in freqs.items() if k not in ("positive", "negative") and v > 0}
    return dict(sorted(core.items(), key=lambda x: x[1], reverse=True))


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="AI Sentiment & Emotion Analyzer", layout="wide")
st.title("🧠 AI Sentiment & Emotion Analyzer")
st.caption(
    "Classify text as positive, negative, or neutral, and detect underlying emotions. "
    "Works on URLs (news/articles/reviews), pasted text, or bulk CSV uploads."
)

transformer_clf = load_transformer()

tab1, tab2, tab3 = st.tabs(["🔗 Analyze a URL", "✍️ Analyze Pasted Text", "📂 Bulk Upload (CSV)"])

# --- Tab 1: URL ---
with tab1:
    url = st.text_input("Paste a news article, blog post, or review page URL")
    if st.button("Analyze URL", key="url_btn"):
        if not url:
            st.warning("Please enter a URL.")
        else:
            with st.spinner("Fetching and analyzing..."):
                text = extract_text_from_url(url)
            if not text:
                st.error("Couldn't extract readable text from that URL. Try a different link, or paste the text manually in the next tab.")
            else:
                with st.expander("Extracted text preview"):
                    st.write(text[:2000] + ("..." if len(text) > 2000 else ""))

                result = classify_sentiment(text, transformer_clf)
                emotions = detect_emotions(text)

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Sentiment")
                    st.metric("Overall Sentiment", result["label"].capitalize())
                    st.write(f"VADER compound score: `{result['vader_compound']}`")
                    if result["transformer_label"]:
                        st.write(f"Transformer model: `{result['transformer_label']}` (confidence {result['transformer_score']})")
                with col2:
                    st.subheader("Emotions detected")
                    if emotions:
                        fig, ax = plt.subplots()
                        ax.bar(list(emotions.keys()), list(emotions.values()), color="#6c5ce7")
                        plt.xticks(rotation=45, ha="right")
                        ax.set_ylabel("Relative frequency")
                        st.pyplot(fig)
                    else:
                        st.write("No strong emotion signals detected.")

# --- Tab 2: Pasted text ---
with tab2:
    text_input = st.text_area("Paste any text (review, comment, paragraph, etc.)", height=200)
    if st.button("Analyze Text", key="text_btn"):
        if not text_input.strip():
            st.warning("Please paste some text.")
        else:
            result = classify_sentiment(text_input, transformer_clf)
            emotions = detect_emotions(text_input)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Sentiment")
                st.metric("Overall Sentiment", result["label"].capitalize())
                st.write(f"VADER compound score: `{result['vader_compound']}`")
                if result["transformer_label"]:
                    st.write(f"Transformer model: `{result['transformer_label']}` (confidence {result['transformer_score']})")
            with col2:
                st.subheader("Emotions detected")
                if emotions:
                    fig, ax = plt.subplots()
                    ax.bar(list(emotions.keys()), list(emotions.values()), color="#00b894")
                    plt.xticks(rotation=45, ha="right")
                    ax.set_ylabel("Relative frequency")
                    st.pyplot(fig)
                else:
                    st.write("No strong emotion signals detected.")

# --- Tab 3: Bulk CSV ---
with tab3:
    st.write(
        "Upload a CSV with a column of text (e.g. Amazon reviews, tweets, headlines). "
        "Optionally include a date column to see sentiment trends over time."
    )
    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)
        st.write("Preview:", df.head())

        text_col = st.selectbox("Which column contains the text to analyze?", df.columns)
        date_col = st.selectbox("Date column (optional, for trend chart)", ["(none)"] + list(df.columns))

        if st.button("Run Bulk Analysis"):
            results = []
            progress = st.progress(0, text="Analyzing...")
            n = len(df)
            for i, row in df.iterrows():
                text = str(row[text_col])
                res = classify_sentiment(text, transformer_clf)
                results.append(res["label"])
                progress.progress(min((i + 1) / n, 1.0), text=f"Analyzing {i+1}/{n}")
            progress.empty()

            df["sentiment"] = results

            st.subheader("Results")
            st.dataframe(df)

            # Overall distribution
            st.subheader("Sentiment distribution")
            counts = df["sentiment"].value_counts()
            fig1, ax1 = plt.subplots()
            ax1.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                    colors=["#00b894", "#d63031", "#636e72"])
            st.pyplot(fig1)

            # Trend over time if date column provided
            if date_col != "(none)":
                try:
                    df["_date"] = pd.to_datetime(df[date_col])
                    trend = df.groupby([pd.Grouper(key="_date", freq="D"), "sentiment"]).size().unstack(fill_value=0)
                    st.subheader("Sentiment trend over time")
                    st.line_chart(trend)
                except Exception:
                    st.warning("Couldn't parse the date column for a trend chart.")

            # Download button
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                "Download results as CSV",
                data=csv_buffer.getvalue(),
                file_name=f"sentiment_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )

st.divider()
st.caption(
    "Engine: VADER lexicon + DistilBERT (SST-2) transformer for sentiment, "
    "NRC Emotion Lexicon for emotion detection. Built for internship sentiment analysis project."
)
