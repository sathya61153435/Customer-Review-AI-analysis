# Customer Review AI Analysis

## Overview
This project performs AI-powered customer review analysis using Natural Language Processing (NLP), sentiment analysis, and LLM-generated business insights.

The system analyzes customer reviews, classifies sentiments, generates visualizations, and produces AI-driven business recommendations using the Groq API.

---

# Features

- Customer sentiment analysis using NLTK VADER
- Word cloud generation for positive and negative reviews
- Sentiment distribution visualization
- AI-generated business insights using Groq LLM
- Customer pain point analysis
- Product improvement recommendations

---

# Tech Stack

- Python
- Pandas
- Matplotlib
- NLTK
- WordCloud
- Groq API
- Prompt Engineering

---

# Project Structure

customer-review-ai-analysis/

│── data/
│   └── reviews.csv
│
│
│   main.py
│
│
│
│── requirements.txt
│── .env
│── README.md

---

# Setup Instructions

## 1. Clone Repository

git clone <your-repository-link>

cd customer-review-ai-analysis

---

## 2. Create Virtual Environment

### Windows

python -m venv venv

Activate environment:

venv\Scripts\activate

---

## 3. Install Dependencies

pip install -r requirements.txt

---

## 4. Configure Groq API

Create a `.env` file:

GROQ_API_KEY=your_api_key

---



## 6. Run Project

python main.py

---

# Functionalities

## Sentiment Analysis
Classifies reviews into:
- Positive
- Negative
- Neutral

## Word Cloud Visualization
Generates:
- All review word cloud
- Positive review word cloud
- Negative review word cloud

## AI Business Insights
Uses Groq LLM to generate:
- Customer issues
- Positive trends
- Product strengths
- Business recommendations
- Improvement suggestions

---

# Skills Demonstrated

- NLP
- Sentiment Analysis
- Prompt Engineering
- AI Business Insights
- Data Visualization
- LLM Integration
- Business Intelligence

---

# Future Improvements

- Dashboard integration
- Real-time review ingestion
- Streamlit deployment
- Topic modeling
- PDF report generation

---

# Author

Jyothi Sathya Vanapamala