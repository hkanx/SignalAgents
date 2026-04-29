# SignalAgents: AI-Powered Brand Intelligence & Automated Issue Resolution


> **Turn customer complaints into actionable insights — automatically.**


SignalAgents is an AI-driven brand monitoring platform that continuously scans public social media and web sources for customer sentiment, detects emerging issues before they escalate, and automates the handoff between customer support, product, and engineering teams. What used to take hours of manual triage now happens in seconds.


---


## Why SignalAgents?


Every day, customers talk about your brand online. Complaints go unnoticed. Bugs get reported on Reddit before they hit your ticket queue. Negative sentiment snowballs.


SignalAgents closes that gap.


| Challenge | SignalAgents Solution |
|-----------|-------------------|
| Customer complaints scattered across Reddit, forums, and review sites | Centralized ingestion with real-time sentiment analysis |
| Manual triage is slow and inconsistent | AI-powered triage automatically routes issues to support or engineering |
| Product teams miss emerging trends | Keyword diagnostics and trend detection surface patterns early |
| Engineering learns about bugs from escalations, not data | Automated Jira intake form creation from complaint analysis |
| No visibility into competitor brand health | Multi-brand analysis enables competitor benchmarking |


---


## Who Is This For?


### Business & Marketing
- Monitor brand health with a single score and trend line
- Track sentiment shifts by topic, keyword, and subreddit
- Generate ready-to-send customer response drafts powered by your knowledge base
- Use competitor analysis mode to benchmark against the gift card and e-commerce landscape


### Product
- Discover emerging issue trends before they become P1s
- View prioritized issue playbooks with severity scores and SLA recommendations
- Build your backlog directly from customer signal data — not guesswork


### Engineering
- Get Jira intake tickets auto-generated from confirmed technical bugs
- See exactly which components (checkout, activation, balance, redemption) are affected
- Criticality scoring (0-10) ensures only real issues hit the queue


### Customer Service
- AI-generated response drafts are informed by your knowledge base articles
- Priority case queue ranks the most urgent complaints for immediate action
- Seamless integration path: when an issue needs engineering help, the Jira intake form is one click away


---


## What It Does


### 1. Brand Sentiment Monitoring
Fetches public data from Reddit (and optionally Bing web search), filters for brand relevance, and runs each post through OpenAI for structured sentiment analysis: sentiment, category, severity, confidence, and a human-readable reason.


### 2. AI-Powered Complaint Triage
An LLM-based triage agent reads each complaint and decides the best course of action:
- **Customer Response** — drafts an empathetic, knowledge-base-informed reply ready for the support team
- **Jira Ticket** — generates a complete D2C Intake form submission for the engineering team, with problem statement, proposed solutions, urgency, and risk assessment


### 3. Automated Jira Intake Form Filling
When a complaint crosses the criticality threshold (default 7.0/10), the system uses browser automation (Playwright) to fill out the actual Jira D2C Intake: Issue Resolution Request Form — including all 13 fields, dropdowns, and rich-text editors. A screenshot is captured and stored for audit trail.


### 4. Multi-Brand & Competitor Analysis
SignalAgents is not locked to a single brand. Change the company name and synonyms in the sidebar, and you can analyze **any brand or competitor** in the gift card, e-commerce, or fintech space. Run it for your competitors to understand their pain points, or benchmark your brand health against the landscape.


---


## Dashboard Tabs — How to Use


### Tab 1: Executive Brand Image Summary
**Audience:** Leadership, Marketing, Product


The top-level view of your brand health at a glance.


- **Brand Image Score** (0-100) — computed from average sentiment across all analyzed posts
- **Sentiment Trend Over Time** — monthly positive/neutral/negative share over the lookback period
- **Sentiment Distribution** — bar chart showing volume by sentiment category
- **Key Metrics** — avg sentiment score, positive %, negative %, net sentiment, and deltas vs. baseline
- **Status Indicator** — "Stable", "Watch", or "At Risk" based on sentiment thresholds
- **Response Playbook** — prioritized P1/P2/P3 action items with recommended SLAs and owner teams


### Tab 2: Customer Response Generator
**Audience:** Customer Service, Marketing


The priority response queue for the most urgent customer complaints.


- **Priority Case Queue** — complaints ranked by severity and sentiment score, most critical first
- **Expandable Case Cards** — each case shows full post text, analysis results, category, and severity
- **AI Response Drafts** — click to generate a ready-to-send response informed by your OpenSearch knowledge base articles
- **KB Article Citations** — see which FAQ or catalog articles were used to craft the response


### Tab 3: Data Analytics
**Audience:** Product, Engineering, Data teams


Deep-dive analytics for issue detection and trend analysis.


- **Issue Diagnosis Table** — extracted topics with mention counts, negative rates, and severity scores
- **Topic Scatter Plot** — visualize issues by sentiment score vs. severity
- **Emerging Risk Keywords** — keywords with disproportionate negative sentiment (lift analysis)
- **Top Positive/Negative Keywords** — word frequency breakdown by sentiment
- **Subreddit Risk View** — which communities have the highest negative rates
- **Raw Data Export** — full analyzed dataset available for download
- **Reddit Ingestion Diagnostics** — API stats, relevance scores, filter breakdown


### Tab 4: Jira Intake Triage
**Audience:** Engineering, Product, Customer Service


The automated bridge from customer complaint to engineering action.


1. **Paste a Reddit URL** and click "Analyze Post"
2. View the **sentiment analysis** (sentiment, category, severity, confidence)
3. See the **AI triage decision** — customer response or Jira ticket — with a **criticality score** (0-10)
4. For Jira tickets: review the **complete intake form preview** (all 13 fields) in a two-column layout
5. Click **"Fill Out Jira Intake Form"** — the system uses browser automation to fill the actual Jira form and captures a screenshot
6. **Submission History** — scroll down to see a persistent log of all previously submitted intake forms with timestamps, metadata, and screenshots


---


## Project Structure


```
SignalAgents/
 app.py                          # Streamlit dashboard (all 4 tabs)
 analyzer.py                     # OpenAI sentiment analysis engine
 triage_agent.py                 # Standalone CLI triage agent
 requirements.txt                # Python dependencies
 .env.example                    # Environment variable template
 data/reviews.json               # Sample review data
 kb/knowledge_base.json          # Knowledge base reference
 triage_history/                 # Persistent screenshots & metadata from Jira submissions
 archive_scripts/                # Archived earlier versions of triage scripts
   triage_agent_archive.py
   jira_client_archive.py
 utils/
   jira_client.py                # Jira REST API + Playwright form automation
   reddit_affiliate_filter.py    # BHN-specific relevance scoring
   keyword_diagnostics.py        # Issue detection, brand health, risk keywords
   response_generator.py         # KB-powered LLM response drafting
   opensearch_kb.py              # AWS OpenSearch knowledge base client
   trend_detection.py            # Negative sentiment spike detection
 .streamlit/config.toml          # Streamlit configuration
```


---


## Setup


### 1. Install dependencies


```bash
cd SignalAgents
pip install -r requirements.txt
playwright install chromium
```


### 2. Create your environment file


```bash
cp .env.example .env
```


Open `.env` and fill in your keys:


| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for sentiment analysis and triage |
| `OPENAI_MODEL` | No | Model name (default: `gpt-4.1-mini`) |
| `REDDIT_USER_AGENT` | No | Reddit API courtesy header (default provided) |
| `BING_API_KEY` | No | Only needed if using Web or Reddit + Web mode |
| `OPENSEARCH_ENDPOINT` | No | AWS OpenSearch endpoint for KB-powered responses |
| `OPENSEARCH_INDEX` | No | OpenSearch index name (default provided) |
| `AWS_REGION` | No | AWS region for OpenSearch (default: `us-west-2`) |
| `AWS_PROFILE` | No | AWS profile name (leave blank for default credential chain) |
| `JIRA_BASE_URL` | For triage | Your Jira Cloud base URL |
| `JIRA_USER_EMAIL` | For triage | Email associated with your Jira API token |
| `JIRA_API_TOKEN` | For triage | Jira API token ([create one here](https://id.atlassian.com/manage-profile/security/api-tokens)) |
| `JIRA_PROJECT_KEY` | For triage | Jira project key (e.g., `GIFT`) |
| `JIRA_ISSUE_TYPE` | No | Issue type for created tickets (default: `Task`) |


### 3. Run the dashboard


```bash
streamlit run app.py
```


### 4. Run the standalone CLI triage agent (optional)


```bash
python triage_agent.py
```

### 5. Run retrieval evaluation harness (optional)

Create an input JSON that follows `data/retrieval_eval_input.schema.json`, then run:

```bash
python scripts/run_retrieval_eval.py \
  --input /path/to/retrieval_eval_input.json \
  --output-json retrieval_eval_report.json \
  --output-md retrieval_eval_summary.md
```

This evaluates `bm25`, `vector`, and `hybrid` retrieval modes at `k=5,10,20` and outputs:
- `recall@k`
- `mrr@k`
- `ndcg@k`

Build a stronger retrieval eval seed set from analyzed rows:

```bash
python scripts/build_retrieval_eval_seed.py \
  --input /path/to/analyzed_rows.json \
  --out-seed data/retrieval_eval_seed.json \
  --out-stats data/retrieval_eval_seed_stats.json \
  --out-summary data/retrieval_eval_seed_summary.md
```

If you are testing with a small sample file, add `--allow-small` to bypass quality-gate blocking.


---


## Important Notes on Rate Limits


Reddit's public JSON API and OpenAI's API both enforce rate limits. If you encounter timeouts or errors:


- **Reddit**: The public endpoint may throttle requests if too many are made in a short window. Wait 30-60 seconds between runs if you see `429` errors or connection timeouts.
- **OpenAI**: Analyzing many posts in a batch can trigger rate limits. Reduce the "Max reviews to analyze" slider in the sidebar if you hit quota errors.
- **General advice**: If you just ran a large analysis, wait a minute or two before starting another one. The app includes retry logic with exponential backoff, but persistent rate limiting may require a brief cooldown.


---


## Multi-Brand & Competitor Analysis


SignalAgents is designed to analyze **any brand** — not just your own. Simply change the company name and synonyms in the sidebar:


- **Your brand**: `Giftcards.com` with synonyms `bhn, blackhawk network, giftcardmall, CashStar, tango card`
- **A competitor**: `Raise.com` with synonyms `raise, raise gift cards, raise.com reviews`
- **A category**: `prepaid visa gift card` with synonyms `visa gift card, mastercard gift card, amex gift card`


This makes SignalAgents a powerful tool for:
- **Competitive benchmarking** — compare sentiment scores and issue themes across brands
- **Market landscape analysis** — understand pain points across the entire gift card / e-commerce category
- **Opportunity identification** — find gaps where competitors are failing that your brand can win


---


## Tech Stack


| Component | Technology |
|-----------|------------|
| Dashboard | Streamlit |
| LLM / AI | OpenAI API (GPT-4.1-mini) |
| Data Sources | Reddit public API, Bing Web Search API |
| Knowledge Base | AWS OpenSearch |
| Browser Automation | Playwright (Chromium) |
| Ticket Management | Jira Cloud REST API + D2C Intake Form |
| Language | Python 3.10+ |


---


## Impact


- **Faster response times** — AI-generated response drafts cut triage-to-reply time from hours to seconds
- **Fewer missed issues** — automated scanning catches complaints that manual monitoring would miss
- **Data-driven prioritization** — severity scoring and trend detection replace gut-feel triage
- **Cross-team alignment** — a single source of truth for Support, Product, and Engineering
- **Scalable across brands** — analyze your brand, competitors, or the entire market with one tool
- **Audit trail** — every Jira submission is timestamped and screenshotted for compliance and tracking
