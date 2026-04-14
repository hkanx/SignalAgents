# SignalAgents: Brand Image Monitor

A simple app that monitors brand sentiment using real Reddit public search data and optional Bing web-search data, then analyzes review text with the OpenAI API.

## Features
- Presets with editable company synonyms
- Reddit ingestion via public JSON endpoint (`https://www.reddit.com/search.json`) using `requests`
- Reddit lookback options: last `1`, `2`, `3`, `4`, or `5` years (cumulative)
- Reddit pagination by page count per term for deeper lookback coverage
- Strict affiliate relevance filtering for Blackhawk Network ecosystem terms
- Optional web-search ingestion via Bing Web Search API
- Source modes: `Reddit`, `Web`, `Reddit + Web`
- Cross-source dedupe (URL/title/text hash)
- OpenAI sentiment analysis via `analyze_review(text)`
- In-memory results only (no database)

## Affiliate Filter Script
Filtering logic lives in a separate reusable script:
- `utils/reddit_affiliate_filter.py`

It includes:
- Include terms 
- Exclude terms for false positives

## Requirements
- Python 3.10+
- OpenAI API key
- Bing Web Search API key only if using `Web` or `Reddit + Web`

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
- optional `REDDIT_USER_AGENT`
- `BING_API_KEY` only if using web mode

## Run
```bash
streamlit run app.py
```

## Sidebar Inputs
- Data source mode: `Reddit`, `Web`, or `Reddit + Web`
- Company name and comma-separated synonyms
- Reddit subreddit, posts-per-term, lookback years, pages-per-term
- Bing web results-per-term (shown only in web-enabled modes)

## Security Notes
- Do not hardcode API keys in code.
- Use environment variables only.
- This project uses real API-returned data and does not fabricate review content.
