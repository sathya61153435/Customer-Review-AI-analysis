import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from groq import Groq
import os
from dotenv import load_dotenv


load_dotenv()
# =========================================================
# GROQ CLIENT
# =========================================================
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# =========================================================
# DOWNLOAD VADER MODEL
# =========================================================
nltk.download('vader_lexicon')

# =========================================================
# LOAD DATASET
# =========================================================
df = pd.read_csv("data/reviews.csv")

# =========================================================
# SELECT REQUIRED COLUMNS
# =========================================================
df = df[['name', 'brand', 'reviews.text', 'reviews.rating']]

# Remove null reviews
df.dropna(subset=['reviews.text'], inplace=True)

# Rename column
df.rename(columns={'reviews.text': 'reviewText'}, inplace=True)

# =========================================================
# SENTIMENT ANALYSIS
# =========================================================
sia = SentimentIntensityAnalyzer()

df['sentiment_score'] = df['reviewText'].apply(
    lambda x: sia.polarity_scores(str(x))['compound']
)

# Sentiment Label Function
def label(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df['sentiment'] = df['sentiment_score'].apply(label)

# =========================================================
# SENTIMENT COUNTS
# =========================================================
print("\nSentiment Counts:\n")
print(df['sentiment'].value_counts())

# =========================================================
# AVERAGE RATING BY SENTIMENT
# =========================================================
print("\nAverage Rating by Sentiment:\n")
print(df.groupby('sentiment')['reviews.rating'].mean())

# =========================================================
# WORD CLOUD - ALL REVIEWS
# =========================================================
all_text = " ".join(df['reviewText'].astype(str))

wordcloud_all = WordCloud(
    width=1000,
    height=500,
    background_color='white'
).generate(all_text)

plt.figure(figsize=(12,6))
plt.imshow(wordcloud_all, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud - All Reviews")
plt.show()

# =========================================================
# WORD CLOUD - POSITIVE REVIEWS
# =========================================================
positive_text = " ".join(
    df[df['sentiment'] == 'Positive']['reviewText'].astype(str)
)

wordcloud_positive = WordCloud(
    width=1000,
    height=500,
    background_color='white'
).generate(positive_text)

plt.figure(figsize=(12,6))
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud - Positive Reviews")
plt.show()

# =========================================================
# WORD CLOUD - NEGATIVE REVIEWS
# =========================================================
negative_text = " ".join(
    df[df['sentiment'] == 'Negative']['reviewText'].astype(str)
)

if len(negative_text) > 0:

    wordcloud_negative = WordCloud(
        width=1000,
        height=500,
        background_color='white'
    ).generate(negative_text)

    plt.figure(figsize=(12,6))
    plt.imshow(wordcloud_negative, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud - Negative Reviews")
    plt.show()

# =========================================================
# SENTIMENT DISTRIBUTION CHART
# =========================================================
df['sentiment'].value_counts().plot(
    kind='bar',
    figsize=(6,4),
    title='Sentiment Distribution'
)

plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# =========================================================
# GENERATE AI BUSINESS INSIGHTS USING GROQ
# =========================================================
def generate_insights(text):

    prompt = f"""
    You are a business analyst.

    Analyze the following customer reviews and provide:

    1. Key customer issues
    2. Positive feedback trends
    3. Product strengths
    4. Business recommendations
    5. Improvement suggestions

    Customer Reviews:
    {text[:4000]}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content

# Take first 50 reviews
sample_reviews = " ".join(df['reviewText'].head(50))

# Generate insights
insights = generate_insights(sample_reviews)

# Print AI Insights
print("\n================ AI BUSINESS INSIGHTS ================\n")
print(insights)

