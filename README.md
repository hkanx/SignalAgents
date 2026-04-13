# SignalAgents: Brand Image Monitor

A simple app that monitors brand sentiment using real Reddit public search data and Bing web-search data, then analyzes review text with the OpenAI API.

## Features
- Presets with editable company synonyms
- Reddit ingestion via public JSON endpoint (`https://www.reddit.com/search.json`) using `requests`
- Web-search ingestion via Bing Web Search API
- Source modes: `Reddit`, `Web`, `Reddit + Web`
- Cross-source dedupe (URL/title/text hash)
- OpenAI sentiment analysis via `analyze_review(text)`
- In-memory results only (no database)
- Dashboard includes:
  - Negative review count
  - Sentiment distribution chart
  - Platform breakdown chart
  - Reviews table with negative rows highlighted

## Requirements
- Python 3.10+
- OpenAI API key
- Bing Web Search API key

## Setup
1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Copy environment template and set your keys:

```bash
cp .env.example .env
```

3. Edit `.env` and set:
- `OPENAI_API_KEY`
- optional `OPENAI_MODEL`
- `BING_API_KEY`
- optional `REDDIT_USER_AGENT`

## Run
```bash
streamlit run app.py
```

## Sidebar Inputs
- Data source mode: `Reddit`, `Web`, or `Reddit + Web`
- Company name and comma-separated synonyms
- Reddit subreddit, time filter, posts-per-term
- Bing web results-per-term

## Security Notes
- Do not hardcode API keys in code.
- Use environment variables only.
- This project uses real API-returned data and does not fabricate review content.
