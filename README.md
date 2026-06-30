# AI Sentiment & Emotion Analyzer

A Streamlit web app for an internship project on sentiment analysis. Lets users
analyze sentiment (positive / negative / neutral) and underlying emotions from
URLs, pasted text, or bulk-uploaded CSV files (e.g. Amazon reviews, tweets,
news headlines).

## How it works

1. **Text extraction** — `trafilatura` pulls clean, readable article/review
   text out of any URL (strips ads, nav bars, boilerplate).
2. **Sentiment classification** — combines two methods for accuracy:
   - **VADER** (lexicon-based): great at handling negation, slang, emojis,
     and punctuation-based emphasis; also the only one of the two that can
     output a genuine "neutral" label.
   - **DistilBERT (SST-2 fine-tune)**: a pretrained transformer model from
     HuggingFace, more accurate on longer, well-formed text. Used to confirm
     or override VADER's positive/negative call when confidence is high.
3. **Emotion detection** — `NRCLex`, built on the NRC Emotion Lexicon, maps
   words to emotions like joy, anger, fear, sadness, trust, anticipation,
   surprise, and disgust.
4. **Trend visualization** — for bulk CSV uploads, the app shows an overall
   sentiment distribution (pie chart) and, if a date column is provided, a
   sentiment trend line chart over time — useful for spotting public opinion
   shifts (e.g. after a product launch or news event).
5. **Export** — results can be downloaded as CSV for further analysis or to
   include in your report's appendix.

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`. The first run
will download the DistilBERT model (~260MB) — this only happens once.

## Using the app

- **Analyze a URL tab** — paste a news article, blog post, or product review
  page link. The app extracts the text and classifies it.
- **Analyze Pasted Text tab** — paste any raw text directly (useful when a
  site blocks scraping, or for short snippets like tweets/comments).
- **Bulk Upload tab** — upload a CSV (e.g. exported Amazon reviews or
  scraped tweets) with a text column, and optionally a date column, to get
  distribution charts, trend lines, and a downloadable results file.

## Suggested data sources for your report

- **Amazon reviews**: Kaggle's "Amazon Product Reviews" datasets (CSV, has
  review text + star ratings you can compare against predicted sentiment).
- **Social media**: Kaggle's "Sentiment140" Twitter dataset, or Reddit data
  via the `praw` library (requires a free Reddit API key).
- **News**: NewsAPI.org (free tier) for headlines/URLs, or Kaggle's "All the
  News" dataset.

## Notes for your report

- This is a **hybrid lexicon + pretrained transformer** approach rather than
  a model trained from scratch — appropriate given the "pretrained, faster,
  more accurate out of the box" priority. You can mention in your
  methodology section that VADER provides interpretable, rule-based
  sentiment (good for explaining *why* a score was given) while the
  transformer adds deep-learning-based accuracy on longer text.
- You can extend this by fine-tuning a transformer on a specific domain
  (e.g. only Amazon reviews) if you want to demonstrate custom model
  training as a "future work" section.
