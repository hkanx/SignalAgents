# SignalAgents: Brand Image Monitor

A simple local Streamlit app that analyzes real review data with the OpenAI API.

## Features
- Loads reviews from `data/reviews.json`
- Runs `analyze_review(text)` for each review using OpenAI
- Stores analysis results in memory only (no database)
- Shows:
  - Negative review count
  - Sentiment distribution chart
  - Platform breakdown chart
  - Reviews table with negative rows highlighted

## Requirements
- Python 3.10+
- OpenAI API key

## Setup
1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Copy environment template and set your key:

```bash
cp .env.example .env
```

Then edit `.env` and set:
- `OPENAI_API_KEY`
- optional `OPENAI_MODEL`

## Run
```bash
streamlit run app.py
```

## Reviews JSON format
`data/reviews.json` must be a JSON array of objects. Each object must include:
- `text` (string)

Recommended fields:
- `platform` (string) or `source` (string)
- `date` (ISO date/time string)
- any additional metadata (e.g. `author`, `rating`)

Example structure (field-only reference):

```json
[
  {
    "text": "...",
    "platform": "...",
    "date": "..."
  }
]
```

Use real data only. This project does not generate sample review content.
