# 🧩 Financial Sentiment Analysis

**Sentiment Dynamics in AAPL Financial News Using FinBERT**

---

## 1️⃣ Research Motivation and Background

### What

This project develops a **data-driven sentiment analysis pipeline** to quantify and interpret **Apple Inc. (AAPL)**’s market mood using **financial news** from the **Benzinga API**.
Both news and historical price data are sourced from Benzinga and processed within a **Delta Lakehouse** architecture on Databricks.
Daily sentiment indicators are generated using **FinBERT**, and their relationship with AAPL’s price and volatility is quantitatively evaluated [4].

### Why

Financial news is a primary channel through which investor sentiment forms and spreads.
Prior research has shown that **text-based sentiment measures** are linked to short-term returns and volatility [1][2][3].
This project applies these insights to Apple’s financial news flow, providing a structured empirical framework to examine **how sentiment co-moves with market performance**.

### How (Research Lineage)

| Concept                               | Prior Evidence                | Adaptation in This Project                                        |
| ------------------------------------- | ----------------------------- | ----------------------------------------------------------------- |
| Market mood predicts price trends     | Bollen et al. (2011) [1]      | Construct daily sentiment indices for AAPL using FinBERT          |
| Firm-level sentiment predicts returns | Li et al. (2017) [2]          | Aggregate FinBERT sentiment scores for AAPL over time             |
| Negative tone increases volatility    | Groth & Muntermann (2011) [3] | Compare sentiment variability with AAPL’s daily return volatility |

---

## 2️⃣ Research Objectives

1. **Ingest** AAPL-related financial news and historical price data from Benzinga.
2. **Curate and store** data in a Delta Lakehouse (Bronze → Silver → Gold).
3. **Generate** sentiment scores using FinBERT [4].
4. **Evaluate** the quantitative relationship between sentiment, returns, and volatility.
5. **Deliver** a reproducible, interpretable sentiment–price analysis framework optimized for Databricks.

---

## 3️⃣ System Architecture

```
┌──────────────────────────────────────────────┐
│ Data Ingestion: Benzinga (news + price data) │
└──────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────┐
│ Bronze Layer (raw JSON)  │
└──────────────────────────┘
            │
            ▼
┌──────────────────────────┐
│ Silver Layer (curated)   │
│ - Deduplication          │
│ - Timestamp alignment    │
│ - Language filtering     │
│ - Ticker linking (AAPL)  │
└──────────────────────────┘
            │
            ▼
┌─────────────────────────────────┐
│ FinBERT Sentiment Inference     │
│ - p_pos, p_neu, p_neg           │
│ - sentiment_score = p_pos–p_neg │
└─────────────────────────────────┘
            │
            ▼
┌──────────────────────────┐
│ Gold Layer (Aggregated)  │
│ - Daily sentiment metrics│
│ - Joined with price data │
└──────────────────────────┘
            │
            ▼
┌────────────────────────────┐
│ Evaluation & Visualization │
│ - Correlation & Co-movement│
└────────────────────────────┘
```

---

## 4️⃣ Data Preparation and Feature Engineering

### 4.1 Ingestion & Curation

* **Data Source:** [Benzinga API](https://www.benzinga.com/apis)

  * Provides both financial news and historical market data for AAPL.

**Bronze Layer:**
Store raw JSON payloads with metadata.

**Silver Layer:**

* Remove duplicates using headline hashes and cosine similarity.
* Normalize timestamps to UTC.
* Filter for English-language text and ticker “AAPL.”

Result: a clean, time-aligned corpus of AAPL news articles.

---

### 4.2 FinBERT Sentiment Inference

* **Model Used:** `ProsusAI/finbert` [4]

  * Pre-trained for financial news sentiment classification (positive, neutral, negative).

**Outputs per document:**
`p_pos`, `p_neu`, `p_neg`
`sentiment_score = p_pos - p_neg`
`confidence = max(p_pos, p_neu, p_neg)`

Applied as a **PySpark or Pandas UDF** across the curated dataset.
Results are stored as `silver.sentiment_docs`.

---

### 4.3 Daily Aggregation

Aggregate FinBERT outputs into daily metrics in the **Gold layer**:

* `sent_mean = mean(sentiment_score)`
* `pos_ratio = mean(1[p_pos > p_neg])`
* `doc_count = count(*)`
* Merge with Benzinga historical prices to compute:

  * `ret_1d = log(P_t / P_{t-1})`
  * `vol_1d = |ret_1d|`

---

## 5️⃣ Sentiment Analysis and Evaluation

### Quantitative Analysis

1. **Correlation and Co-movement Analysis:**
   Quantify the relationship between daily FinBERT sentiment metrics (`sent_mean`, `pos_ratio`, `doc_count`) and AAPL’s market indicators (`ret_1d`, `vol_1d`).
   Compute **Pearson** and **Spearman** correlations [1][2][3] to assess the direction and strength of association.
   Visualize results using **time-series overlays** and **scatter plots** of sentiment versus returns and volatility.
   *Purpose:* establish the baseline quantitative linkage between financial news sentiment and market performance.

---

## 6️⃣ Implementation Roadmap

| Phase                    | Duration | Deliverable                                              |
| ------------------------ | -------- | -------------------------------------------------------- |
| **1. Setup & Ingestion** | 1 week   | Bronze tables with Benzinga news and price data          |
| **2. Curation (Silver)** | 1 week   | Clean, deduplicated AAPL news dataset                    |
| **3. FinBERT Inference** | 1 week   | Document-level sentiment scores stored in Delta          |
| **4. Daily Aggregation** | 0.5 week | Gold-layer daily sentiment metrics joined with prices    |
| **5. Evaluation**        | 1 week   | Correlation and co-movement analysis with visualizations |
| **6. Reporting**         | 1 week   | Final report and reproducible Databricks notebooks       |

---

## 7️⃣ Expected Contributions

### Academic

* Provides empirical evidence of **news-based sentiment–price relationships** for a major technology firm [1][2][3].
* Demonstrates how FinBERT sentiment co-moves with market data in a reproducible setting.

### Engineering

* Implements an **end-to-end sentiment pipeline** using Delta Lake and FinBERT on Databricks Free Edition.
* Provides modular PySpark workflows for financial text analytics.

### Educational

* Offers a compact case study on **financial NLP and market correlation analysis**.
* Highlights interpretability through clear visualizations of sentiment–price co-movement.

---

## 8️⃣ Reference Mapping

| Component                    | Key Source                    | Relevance                                           |
| ---------------------------- | ----------------------------- | --------------------------------------------------- |
| Sentiment index construction | Bollen et al. (2011) [1]      | Links aggregate sentiment and market movement       |
| Firm-level aggregation       | Li et al. (2017) [2]          | Company-level sentiment predicting returns          |
| Text–volatility relation     | Groth & Muntermann (2011) [3] | Negative tone associated with short-term volatility |
| FinBERT model                | Araci (2019) [4]              | FinBERT for financial news sentiment classification |

---

## 9️⃣ IEEE-Style References

[1] J. Bollen, H. Mao, and X. Zeng, “Twitter mood predicts the stock market,” *Journal of Computational Science*, vol. 2, no. 1, pp. 1–8, 2011.
[2] B. Li, K. C. C. Chan, C. Ou, and R. Sun, “Discovering public sentiment in social media for predicting stock movement of publicly listed companies,” *Information Systems*, vol. 69, pp. 81–92, 2017.
[3] S. S. Groth and J. Muntermann, “An intraday market risk management approach based on textual analysis,” *Decision Support Systems*, vol. 50, no. 4, pp. 680–691, 2011.
[4] D. T. Araci, “FinBERT: Financial sentiment analysis with pre-trained language models,” *Unpublished manuscript*, University of Amsterdam, 2019.

---