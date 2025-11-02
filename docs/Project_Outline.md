# 🧩 Project Title

**Sentiment Dynamics in AAPL Financial Communications Using FinBERT and Text Data**

---

## 1️⃣ Research Motivation and Background

### What

The project builds a **data-driven sentiment analysis pipeline** that quantifies and interprets **Apple Inc. (AAPL)**’s market mood through textual data such as:

* **Financial news** (e.g., Reuters, CNBC)
* **Official press releases** (Apple newsroom)
* **Earnings-call transcripts**

It integrates a **Delta Lakehouse** for data curation, and **FinBERT-based sentiment analysis** to capture day-to-day fluctuations in market sentiment. The project focuses on quantitative and qualitative evaluation of sentiment outputs.

---

### Why

Academic research shows that **textual sentiment and tone** reflect investor psychology and convey information not visible in price or volume data.

* **Aggregate sentiment** predicts short-term returns and volatility (Bollen et al., 2011).
* **Firm-level textual mood** signals company-specific optimism or concern (Li et al., 2017).
* **Official disclosures and tone** convey fundamental outlook (Feuerriegel & Gordon, 2018).

This project operationalizes those insights to **measure sentiment**, **relate it to market events**, and **interpret language patterns** that shape investor perception of Apple.

---

### How (research lineage → our build)

| Concept                                         | Empirical evidence                                                                          | Our adaptation                                                                                  |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Market mood predicts price dynamics**         | Bollen, Mao & Zeng (2011) — aggregate Twitter sentiment Granger-causes DJIA movement.       | Build daily “market mood” indices for AAPL using FinBERT sentiment from news, press, and calls. |
| **Firm-level sentiment matters**                | Li et al. (2017) — company-specific social sentiment predicts daily returns.                | Aggregate FinBERT scores at firm level and analyze temporal and event-driven patterns.          |
| **Textual tone signals risk**                   | Groth & Muntermann (2011) — negative tone increases volatility.                             | Compare daily sentiment variance with AAPL price volatility.                                    |
| **Official text sources add fundamentals**      | Feuerriegel & Gordon (2018) — regulatory disclosures hold long-term predictive information. | Include Apple’s press releases and earnings-call transcripts in data feed.                      |
| **Financial lexicons improve interpretability** | Loughran & McDonald (2011) — domain dictionaries outperform generic sentiment lists.        | Apply LM lexicon for qualitative tone exploration and keyword interpretation.                   |

---

## 2️⃣ Research Objectives

1. **Construct** a text-ingestion and curation system for AAPL-related financial documents.
2. **Generate** daily sentiment indicators using FinBERT and analyze their dynamics over time.
3. **Quantify** relationships between sentiment indices and simple market metrics (returns, volatility).
4. **Interpret** linguistic patterns explaining sentiment extremes and tone shifts.
5. **Provide** an empirical, reproducible sentiment-monitoring framework that can later feed forecasting models.

---

## 3️⃣ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Data Ingestion                      │
│  Topics: news.raw | press.raw | earnings.raw | prices.daily │
└─────────────────────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────┐
│ Bronze Layer (Delta)          │
│ Raw JSON payloads, metadata   │
└───────────────────────────────┘
            │
            ▼
┌───────────────────────────────┐
│ Silver Layer (Curated)        │
│ - Deduplication               │
│ - Timestamp alignment         │
│ - Ticker linking (AAPL)       │
│ - Language filtering          │
└───────────────────────────────┘
            │
            ▼
┌────────────────────────────────┐
│ FinBERT Sentiment Analysis     │
│ - p_pos, p_neu, p_neg          │
│ - sentiment_score (p_pos-p_neg)│
│ - confidence metrics           │
└────────────────────────────────┘
            │
            ▼
┌───────────────────────────────┐
│ Gold Layer (Sentiment Metrics)│
│ Daily aggregation: mean, std, │
│ pos_ratio, doc_count          │
└───────────────────────────────┘
            │
            ▼
┌───────────────────────────────┐
│ Analysis & Visualization      │
│ - Quantitative: correlations  │
│ - Qualitative: keywords, tone │
│ - Event timeline plots        │
└───────────────────────────────┘
```

---

## 4️⃣ Data Preparation and Feature Engineering

### 4.1 Ingestion & Curation

* **Raw Data:** financial news (Alpaca? Bezinga? MarketAux), press releases (Alpaca? Bezinga? MarketAux), earnings-call transcripts (Alpaca? Bezinga? MarketAux), and price data (Alpaca? Yahoo Finance?).
* **Bronze layer:** stores raw messages with metadata.
* **Silver layer:**

  * Remove duplicates using headline hashes and cosine similarity.
  * Align timestamps to publication time (UTC).
  * Filter for English-language text and link ticker “AAPL.”
  * Compute optional *novelty score* to identify repeated narratives.
* *Motivation:* establishes clean, time-synchronized textual data, per Li et al. (2017) and Groth & Muntermann (2011).*

### 4.2 FinBERT Sentiment Inference

* **Model:** `ProsusAI/finbert`[[6]](#9️⃣-ieee-style-references) (News Analysis)  `yiyanghkust/finbert-tone`[[7]](#9️⃣-ieee-style-references) (Press Releases and earnings calls) - Hugging Face.
* **Output per document:**

  * `p_pos`, `p_neu`, `p_neg`
  * `sentiment_score = p_pos - p_neg`
  * `confidence = max(p_pos, p_neu, p_neg)`
* **Application:** executed as PySpark UDF over Silver layer; results stored in `silver.sentiment_docs`.
* *Grounded in Bollen et al. (2011) for sentiment-index construction and Li et al. (2017) for firm-level aggregation.*

### 4.3 Daily Aggregation

Aggregate FinBERT outputs to trading-day level (`gold.sentiment_daily`):

* `sent_mean = mean(sentiment_score)`
* `sent_std = std(sentiment_score)`
* `pos_ratio = mean(1[p_pos > p_neg])`
* `doc_count = count(*)`
* Merge with AAPL price data to attach `ret_1d` and `vol_1d = |ret_1d|`.

---

## 5️⃣ Sentiment Analysis and Evaluation

### Quantitative Analysis

1. **Descriptive statistics:** mean, variance, and distribution of sentiment scores.
2. **Correlation analysis:** Pearson/Spearman between sentiment measures and market returns/volatility.
3. **Event studies:** sentiment and volatility behavior in ±3-day windows around major events (earnings releases, product launches).
4. **Time-series visualization:** sentiment mean and doc volume overlaid on price trends.
5. **Exploratory Granger test:** check if sentiment leads next-day returns.

### Qualitative Analysis

1. **Keyword extraction:** identify top tokens driving positive/negative classifications.
2. **Word clouds / topic clusters:** reveal themes behind sentiment spikes (e.g., “supply issues,” “record demand”).
3. **Tone interpretation:** apply Loughran–McDonald categories to selected texts for contextual understanding.
4. **Narrative timeline:** annotate sentiment surges with corporate or macro events.

---

## 6️⃣ Implementation Roadmap

| Phase                                      | Duration  | Key Deliverables                                    |
| ------------------------------------------ | --------- | --------------------------------------------------- |
| **1. Setup & Ingestion**                   | 1 week    | Bronze tables with sample AAPL feeds. |
| **2. Curation (Silver)**                   | 1 week    | Deduplicated, aligned, ticker-linked text dataset.  |
| **3. FinBERT Inference**                   | 1 week    | Document-level sentiment scores stored in Delta.    |
| **4. Daily Aggregation**                   | 0.5 week  | `gold.sentiment_daily` table of metrics.            |
| **5. Quantitative & Qualitative Analysis** | 1.5 weeks | Correlation stats, event plots, word clouds.        |
| **6. Reporting**                           | 1 week    | Findings report, visuals, reproducible notebooks.   |

---

## 7️⃣ Expected Contributions

### Academic

* Demonstrates sentiment quantification consistent with behavioral finance literature.
* Provides empirical evidence on how **financial text tone varies across events** and its linkage with volatility.
* Offers a reproducible, intermediate framework that can extend to forecasting tasks.

### Engineering

* Implements **data-to-sentiment** pipeline (Batch Data → Delta → FinBERT).
* Delivers reusable PySpark FinBERT inference module and aggregation scripts.

### Educational

* Enables students and researchers to explore **text analytics before prediction**, emphasizing interpretability.
* Provides visual, explainable insights connecting language to market context.

---

## 8️⃣ Reference Mapping (for deeper reading)

| Pipeline Component           | Principal Source                | What to Read There                           |
| ---------------------------- | ------------------------------- | -------------------------------------------- |
| Sentiment index construction | **Bollen et al. (2011)**        | Mood tracking and Granger-causality design.  |
| Firm-level aggregation       | **Li et al. (2017)**            | Company-specific sentiment feature creation. |
| Text-volatility relation     | **Groth & Muntermann (2011)**   | Negative tone and short-term risk links.     |
| Disclosures as textual data  | **Feuerriegel & Gordon (2018)** | Incorporating official statements.           |
| Tone interpretation          | **Loughran & McDonald (2011)**  | Financial tone categories and dictionary.    |

---

## 9️⃣ IEEE-Style References

[1] J. Bollen, H. Mao, and X. Zeng, “Twitter mood predicts the stock market,” *Journal of Computational Science*, vol. 2, no. 1, pp. 1–8, 2011.
[2] B. Li, K. C. C. Chan, C. Ou, and R. Sun, “Discovering public sentiment in social media for predicting stock movement of publicly listed companies,” *Information Systems*, vol. 69, pp. 81–92, 2017.
[3] S. S. Groth and J. Muntermann, “An intraday market risk management approach based on textual analysis,” *Decision Support Systems*, vol. 50, no. 4, pp. 680–691, 2011.
[4] S. Feuerriegel and J. Gordon, “Long-term stock index forecasting based on text mining of regulatory disclosures,” *Decision Support Systems*, vol. 112, pp. 88–97, 2018.
[5] T. Loughran and B. McDonald, “When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks,” *Journal of Finance*, vol. 66, no. 1, pp. 35–65, 2011.
[6] [Araci, D. T. (2019). FinBERT: Financial sentiment analysis with pre-trained language models [Unpublished manuscript, University of Amsterdam]](https://arxiv.org/pdf/1908.10063)
[7] [Y. Yang, M. C. S. Uy, and A. Huang, “FinBERT: A Pretrained Language Model for Financial Communications,” Findings of the Association for Computational Linguistics: EMNLP 2020, pp. 2732–2740, 2020, doi: 10.18653/v1/2020.findings-emnlp.244.](https://arxiv.org/pdf/2006.08097)

---

### ✅ Summary of Intellectual Scope

These six works collectively support:

* **Behavioral signal theory** — Bollen et al.; Li et al.
* **Text-risk connection** — Groth & Muntermann.
* **Official disclosure analysis** — Feuerriegel & Gordon.
* **Interpretability via domain tone** — Loughran & McDonald.