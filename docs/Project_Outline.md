# Financial Sentiment Analysis

**Sentiment Dynamics in AAPL Financial Communications Using FinBERT and Text Data**

---

## 1. Research Motivation and Background

### What

This project develops a **data-driven sentiment analysis pipeline** to quantify and interpret **Apple Inc. (AAPL)**’s market mood from financial text data, including:

* **Financial news**
* **Official press releases**
* **Earnings-call transcripts**

Using **Benzinga API** for textual data and **Yahoo Finance** for price data, the system integrates a **Delta Lakehouse** for data curation and employs **FinBERT** to generate daily sentiment indicators linked to AAPL’s market behavior [5][6].

### Why

Behavioral finance research shows that **textual tone** captures investor sentiment and can anticipate market movements.
Aggregate and firm-level sentiment have been shown to reflect market psychology and influence asset returns [1][2][3].
By systematically measuring sentiment from company-related communications, this project evaluates how public tone and narratives correlate with AAPL’s short-term returns and volatility.

### How (Research Lineage)

| Concept                               | Prior Evidence                                                 | Adaptation in This Project                                |
| ------------------------------------- | -------------------------------------------------------------- | --------------------------------------------------------- |
| Market mood predicts price dynamics   | Bollen et al. (2011) [1]: aggregate mood leads DJIA trends     | Construct daily sentiment indices for AAPL using FinBERT  |
| Firm-level tone matters               | Li et al. (2017) [2]: company-level sentiment predicts returns | Aggregate FinBERT sentiment for AAPL over time            |
| Negative tone signals risk            | Groth & Muntermann (2011) [3]                                  | Compare sentiment variability with daily price volatility |
| Official disclosures add fundamentals | Feuerriegel & Gordon (2018) [4]                                | Include AAPL press releases and earnings-call transcripts |

---

## 2. Research Objectives

1. **Ingest** AAPL-related text data from Benzinga and market data from Yahoo Finance.
2. **Curate and store** text in a structured Delta Lakehouse (Bronze → Silver → Gold).
3. **Generate** sentiment scores using FinBERT [5][6] for financial news, press releases, and earnings transcripts.
4. **Evaluate** how daily sentiment correlates with AAPL’s returns and volatility.
5. **Deliver** a reproducible sentiment–price evaluation framework on Databricks.

---

## 3. System Architecture

```
┌──────────────────────────────────────────────────┐
│ Data Ingestion: Benzinga (news, press, earnings) │
│               + Yahoo Finance (price data)       │
└──────────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────┐
│ Bronze Layer (raw JSON)  │
└──────────────────────────┘
            │
            ▼
┌───────────────────────────┐
│ Silver Layer (curated)    │
│ - Deduplication           │
│ - Timestamp alignment     │
│ - Language filtering      │
│ - Ticker linking (AAPL)   │
└───────────────────────────┘
            │
            ▼
┌─────────────────────────────────┐
│ FinBERT Sentiment Inference     │
│ - p_pos, p_neu, p_neg           │
│ - sentiment_score = p_pos–p_neg │
└─────────────────────────────────┘
            │
            ▼
┌───────────────────────────┐
│ Gold Layer (Aggregated)   │
│ - Daily sentiment metrics │
│ - Joined with price data  │
└───────────────────────────┘
            │
            ▼
┌─────────────────────────────┐
│ Evaluation & Visualization  │
│ - Correlation & Event Study │
└─────────────────────────────┘
```

---

## 4. Data Preparation and Feature Engineering

### 4.1 Ingestion & Curation

* **Text Data Source:** [Benzinga API](https://www.benzinga.com/apis)

  * Financial news, press releases, and earnings-call transcripts for AAPL.
* **Price Data Source:** [Yahoo Finance](https://finance.yahoo.com/quote/AAPL/history)

  * Daily OHLCV and adjusted closing prices.

**Bronze Layer:**
Raw JSON payloads with metadata stored as Delta tables.

**Silver Layer:**

* Deduplicate using headline hashes and cosine similarity.
* Normalize timestamps to UTC.
* Filter for English text.
* Tag all entries with ticker “AAPL.”

Result: a clean, time-synchronized corpus of AAPL-related text suitable for sentiment inference.

---

### 4.2 FinBERT Sentiment Inference

* **Models Used:**

  * [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) → financial news [5]
  * [yiyanghkust/finbert-tone](https://huggingface.co/yiyanghkust/finbert-tone) → press releases and earnings transcripts [6]

**Outputs per document:**
`p_pos`, `p_neu`, `p_neg`
`sentiment_score = p_pos - p_neg`
`confidence = max(p_pos, p_neu, p_neg)`

Inference is applied as a **PySpark or Pandas UDF** on the Silver layer, storing results as `silver.sentiment_docs`.

---

### 4.3 Daily Aggregation

Aggregate document-level outputs to trading-day metrics in the **Gold layer**:

* `sent_mean = mean(sentiment_score)`
* `pos_ratio = mean(1[p_pos > p_neg])`
* `doc_count = count(*)`
* Merge with Yahoo Finance data to calculate:

  * `ret_1d = log(P_t / P_{t-1})`
  * `vol_1d = |ret_1d|`

---

## 5. Sentiment Analysis and Evaluation

### Quantitative Analysis

1. **Correlation and Co-movement Analysis:**
   Quantify the relationship between daily FinBERT sentiment metrics (`sent_mean`, `pos_ratio`, `doc_count`) and AAPL’s market indicators (`ret_1d`, `vol_1d`).
   Compute **Pearson** and **Spearman** correlations [1][2][3] to measure direction and strength of association.
   Visualize co-movement using **time-series overlays** and **scatter plots** of sentiment vs. returns.
   *Purpose:* establish the baseline quantitative linkage between sentiment and market behavior.

2. **Event-Window Evaluation:**
   Examine sentiment and price behavior around major Apple events such as **earnings announcements** and **product launches** [4].
   Define **±3-day windows** around each event, compute changes in mean sentiment and cumulative abnormal returns, and annotate key events with short contextual notes.
   *Purpose:* identify whether sentiment shifts coincide with, precede, or lag market reactions.

---

## 6. Implementation Roadmap

| Phase                    | Duration | Deliverable                                           |
| ------------------------ | -------- | ----------------------------------------------------- |
| **1. Setup & Ingestion** | 1 week   | Bronze tables with Benzinga and Yahoo Finance data    |
| **2. Curation (Silver)** | 1 week   | Clean, deduplicated, ticker-linked text dataset       |
| **3. FinBERT Inference** | 1 week   | Document-level sentiment scores in Delta              |
| **4. Daily Aggregation** | 0.5 week | Gold-layer daily sentiment metrics joined with prices |
| **5. Evaluation**        | 1 week   | Correlation and event-window analyses, visualizations |
| **6. Reporting**         | 1 week   | Final report and reproducible Databricks notebooks    |

---

## 7. Expected Contributions

### Academic

* Provides empirical evidence of the **sentiment–price relationship** for a major technology firm [1][2][3].
* Demonstrates how FinBERT-based sentiment metrics align with financial market behavior [5][6].

### Engineering

* Implements a complete **data-to-sentiment pipeline** using Delta Lake on Databricks Free Edition.
* Includes reusable PySpark and FinBERT processing modules.

### Educational

* Offers a reproducible case study on **financial NLP integration with market data**.
* Emphasizes interpretability and clear sentiment–price linkage.

---

## 8. Reference Mapping

| Component                    | Key Source                      | Relevance                                                     |
| ---------------------------- | ------------------------------- | ------------------------------------------------------------- |
| Sentiment index construction | Bollen et al. (2011) [1]        | Established link between social sentiment and market movement |
| Firm-level aggregation       | Li et al. (2017) [2]            | Company-specific sentiment predicting returns                 |
| Text–volatility relation     | Groth & Muntermann (2011) [3]   | Negative tone associated with short-term volatility           |
| Disclosure analysis          | Feuerriegel & Gordon (2018) [4] | Official communications’ predictive value                     |
| FinBERT model – news         | Araci (2019) [5]                | FinBERT for financial sentiment analysis                      |
| FinBERT model – tone         | Yang et al. (2020) [6]          | FinBERT for corporate and earnings text tone                  |

---

## 9. References

[1] J. Bollen, H. Mao, and X. Zeng, “Twitter mood predicts the stock market,” *Journal of Computational Science*, vol. 2, no. 1, pp. 1–8, 2011.
[2] B. Li, K. C. C. Chan, C. Ou, and R. Sun, “Discovering public sentiment in social media for predicting stock movement of publicly listed companies,” *Information Systems*, vol. 69, pp. 81–92, 2017.
[3] S. S. Groth and J. Muntermann, “An intraday market risk management approach based on textual analysis,” *Decision Support Systems*, vol. 50, no. 4, pp. 680–691, 2011.
[4] S. Feuerriegel and J. Gordon, “Long-term stock index forecasting based on text mining of regulatory disclosures,” *Decision Support Systems*, vol. 112, pp. 88–97, 2018.
[5] D. T. Araci, “FinBERT: Financial sentiment analysis with pre-trained language models,” *Unpublished manuscript*, University of Amsterdam, 2019.
[6] Y. Yang, M. C. S. Uy, and A. Huang, “FinBERT: A Pretrained Language Model for Financial Communications,” *Findings of the Association for Computational Linguistics: EMNLP 2020*, pp. 2732–2740, 2020.

---
