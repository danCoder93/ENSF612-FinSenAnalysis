# **Sentiment-Driven AAPL Price Forecasting**

A GPU-Accelerated Lakehouse Pipeline for Financial News Processing, FinBERT Sentiment Modeling, and Short-Horizon Stock Prediction

**D. Shahid, S. Kazi, M. Dzukou, and M. Senanayake**  
Department of Electrical and Software Engineering  
University of Calgary, Calgary, AB, Canada

---

## **Abstract**

This repository presents a fully reproducible, GPU-accelerated financial news sentiment analysis and price-direction prediction pipeline deployed on Databricks Serverless GPU. AAPL news articles and market data are ingested into a Delta Lakehouse workflow and processed through an advanced natural language processing (NLP) stack consisting of HTML normalization, text preprocessing, entity-aware sentence segmentation, and neural coreference resolution. FinBERT is applied at the sentence level to generate financial sentiment features, which are averaged across dates and combined with technical indicators (normalized close, RSI, OBV differentials) to construct a supervised learning dataset for 5-day price-direction forecasting.

An XGBoost classifier, trained with and without sentiment features, is used to evaluate whether FinBERT-derived sentiment improves predictive accuracy. Sentiment analysis does not improve out-of-sample accuracy because of sensitivity to tone and the presence of multiple tickers in the same article, indicating that sentiment provides contextual—but noisy—information. This repository provides the complete Databricks implementation, project documentation, and experimental workflow.

---

## **1. Introduction**

Financial markets are influenced by a complex interplay of fundamentals, investor psychology, macroeconomic signals, and news flow. Modern transformer models such as FinBERT have made it possible to quantify the sentiment embedded in financial text with high fidelity. Meanwhile, platforms like Databricks Serverless GPU enable cost-efficient large-scale NLP pipelines traditionally restricted to specialized compute environments.

This project investigates a focused research question derived from the original outline :

> **Does financial news sentiment help predict short-horizon AAPL price direction when combined with traditional technical indicators?**

To answer this, we built a cloud-scale sentiment analysis and prediction pipeline optimized for Databricks GPU environments. The end-to-end system includes:

* Large-scale ingestion of AAPL news
* HTML and markup normalization
* Neural coreference resolution (FCoref)
* Entity-aware sentence segmentation (spaCy)
* FinBERT sentiment scoring
* Technical indicator engineering
* XGBoost binary classification
* Robust experiment evaluation

---

## **2. System Overview**

The pipeline follows the Bronze → Silver → Gold data engineering paradigm also described in the project report :

```
                   ┌──────────────────────────────────┐
                   │   Benzinga News & Alpaca Prices  |
                   |     (src/RawDataDownload.ipynb)  │
                   └───────────────┬──────────────────┘
                                   ▼
                      ┌───────────────────────────┐
                      │   Bronze (Raw JSON)       |
                      |    (DataCataloging.ipynb) │
                      └───────────────────────────┘
                                   ▼
              ┌──────────────────────────────────────────┐
              │  Silver (Clean, Normalized, Deduped)     │
              │  - HTML stripping                        |
              |  - Text Preprocessing                    │
              │  - Timestamp normalization               │
              │  - Sentence segmentation (spaCy)         │
              │  - Coreference resolution (FCoref)       │
              └──────────────────────────────────────────┘
                                   ▼
          ┌────────────────────────────────────────────────────────────────────┐
          │ Gold (ML Ready)                                                    │
          │ - FinBERT sentiment (title/body)                                   │
          │ - Technical indicators (RSI, normalized OBV diff, normalized close)│
          │ - 5-day price-direction labels                                     │
          └────────────────────────────────────────────────────────────────────┘
                                   ▼
                ┌────────────────────────────────────────┐
                │    XGBoost Classifier (GPU-enabled)    │
                └────────────────────────────────────────┘
```

---

## **3. Dataset and Preprocessing**

### **3.1 Data Sources**

The following data sources are used:

* Benzinga AAPL Financial News
  * Titles, teasers, bodies, and timestamps
* Alpaca AAPL Historical Market Data
  * Open, high, low, close, volume

This is consistent with the project’s data definition in the proposal and report.

### **3.2 Timestamp Normalization and grouping**

* Converting timestamps data types
* Grouping the title, teaser, and body based on date

### **3.3 Text Cleaning and Normalization**

Using Textacy and BeautifulSoup:

* Remove HTML tags, emojis, URLs, boilerplate phrases
* Normalize punctuation, bullet characters, whitespace

### **3.4 Coreference Resolution (FCoref)**

FCoref GPU-enabled model ensures pronouns such as “it”, “the company”, “the tech giant” correctly resolve to Apple.

### **3.5 Sentence Segmentation**

Using spaCy:

* Segment documents into sentences
* Retain sentences referencing Apple explicitly or via ORG/PRODUCT NER tags

---

## **4. Sentiment Modeling with FinBERT**

### **4.1 Model**

* **ProsusAI/finbert**
  * Designed specifically for financial news sentiment classification.

### **4.2 Output per Sentence**

* Positive probability
* Neutral probability
* Negative probability

Daily aggregates are then averaged for title, teaser, and body.

---

## **5. Feature Engineering**

Technical indicators from the market data include, calculated:

* RSI (14-day)
* On-Balance Volume (OBV) differential
* Normalized close
* 5-day price direction (binary target)

---

## **6. Machine Learning Model**

### **6.1 XGBoost Classifier**

Two models are trained:

1. Baseline: Technical indicators only
2. Sentiment-Enhanced: Technical + FinBERT features

Train/test split uses first 80% as training and last 20% of time series, avoiding leakage.

---

## **7. Results**

### **7.1 Accuracy**

| Model                 | Train Accuracy | Test Accuracy |
| --------------------- | -------------- | ------------- |
| Technical Only        | 86.20%         | 52.95%        |
| Technical + Sentiment | 94.08%         | 52.46%        |

### **7.2 Interpretation**

* Testing accuracy remains near random chance (≈53%)
* Sentiment increases training accuracy but does not improve real-world predictive power
* Train–test gap increases slightly → sentiment increases overfitting
* Price direction at a 5-day horizon remains highly noisy

---

## **8. Discussion and Limitations**

The project report highlights several constraints:

* Short-term market movements are intrinsically noisy
* Daily sentiment may be too coarse
* AAPL is highly efficient; news-based edges decay rapidly
* Classical models (XGBoost) may not fully capture temporal structures
* Transformer embeddings, multi-day smoothing, or volatility modeling may be required for stronger results
* Sentiment analysis is sensitive to textual tone and the presence of multiple entities Example, a news has a general negative tone for Spotify but alternatively, it is a positive statement from Apple's perspective. Finbert was unable to distinguish sentiment from each entity's perspective. A NER based finbert sentiment analysis might improve overall prediction accuracy

More complicated neural models like TCN, TFT, or hybrid models might improve the overall prediction

---

## **9. Usage Guide**

This repository contains all code needed to reproduce the pipeline on paid version of Databricks Serverless GPU.

### **9.1 Prerequisites**

#### **Databricks Workspace**

* Databricks Runtime 14.x or later
* Access to GPU Serverless A10 with AI v4 Environment
* DBFS Volume enabled

### **Libraries Required**

Your cluster must have:

* `transformers`
* `torch` (GPU-enabled)
* `xgboost` (GPU-enabled)
* `spacy` and English model
* `beautifulsoup4`
* `textacy`
* `fastcoref`
* `ta`

These packages are already adding to be installed via `%pip`.

---

### **9.2 Running the Pipeline**

1. Upload the notebook `FinSenAnalysis.ipynb` to Databricks
2. Attach to a GPU serverless cluster
3. Run the Bronze ingestion cells
4. Run Silver cleaning and NLP pipeline (GPU recommended)
5. Run FinBERT inference
6. Generate the Gold ML dataset
7. Train/evaluate XGBoost models

Each stage is fully contained and can be re-run independently.

---

### **9.3 Compute Requirements**

#### Minimum Requirements

* GPU Serverless with A10
* Adequate capacity for FinBERT batch inference

#### Runtime Expectations

You should be able to reproduce full experiments in under 60 minutes.

---

## **10. Repository Structure**

```
/notebooks
    RawDataDownload.ipynb (For downloading data from APIs)
    DataCataloging.ipynb (For storing data into Volume Catalog from data/*)
    FinSenAnalysis.ipynb (For running the experiment)

/data
    aapl_news.json
    aapl_price.json

/docs
    612-finAnalysis-project-report.pdf
    Project_Outline.md

README.md
```

---

## **11. Conclusion**

This project delivers a robust, sentiment analysis pipeline aligned with modern financial NLP research. While sentiment does not strengthen short-horizon predictions for AAPL, it provides contextual signals not captured by price data alone. More robust NER based sentiment analysis and expressive deep temporal models may uncover richer structure in future work—especially if paired with volatility, macroeconomic features, or higher-frequency news.

The repository is structured for clarity and reproducibility and can serve both as a learning resource and a foundation for more advanced financial ML systems.

---

## **12. References**

[1] D. T. Araci, “FinBERT: Financial sentiment analysis with pre-trained language models,” *Prosus AI Tech Blog*, 2019. [Online]. Available: <https://medium.com/prosus-ai-tech-blog/finbert-financial-sentiment-analysis-with-bert-b277a3607101>

[2] ProsusAI, “FinBERT: Financial Sentiment Analysis,” *Hugging Face*, 2020. [Online]. Available: <https://huggingface.co/ProsusAI/finbert>

[3] S. Otmazgin, A. Cattan, and Y. Goldberg, “F-COREF: Fast, Accurate and Easy to Use Coreference Resolution,” *arXiv preprint arXiv:2209.04280*, 2022. [Online]. Available: <https://arxiv.org/pdf/2209.04280>

[4] XGBoost Developers, “XGBoost Documentation,” *xgboost.readthedocs.io*, 2024. [Online]. Available: <https://xgboost.readthedocs.io/en/stable/tutorials/index.html>

[5] B. DeWilde, “Textacy: NLP Toolkit for Python,” *textacy.readthedocs.io*, 2023. [Online]. Available: <https://textacy.readthedocs.io/en/latest/>

[6] “spaCy Sentencizer,” *spaCy Library*, 2024. [Online]. Available: <hhttps://spacy.io>

[7] “Beautiful Soup Documentation,” *beautiful-soup-4.readthedocs.io*, 2023. [Online]. Available: <https://beautiful-soup-4.readthedocs.io/en/latest/>

[8] Q. Zeng and T. Jiang, “Financial Sentiment Analysis Using FinBERT with Application in Predicting
Stock Movement” *arXiv preprint arXiv:2306.02136*, 2023. [Online]. Available: <https://arxiv.org/pdf/2306.02136>
