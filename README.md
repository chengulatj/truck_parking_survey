
# Truck Parking System (TPAS) Survey Analysis  
### Table of Contents

1. [Project Overview](#project-overview)  
   - [Features](#features)

2. [Getting Started](#getting-started)  
   - [Prerequisites](#prerequisites)  
   - [How to Use](#how-to-use)  
   - [Dependencies](#dependencies)

3. [Sentiment Analysis Module](#sentiment-analysis-module)  
   - [Load Pre-trained Model](#1-load-pre-trained-model)  
   - [Define Sentiment Prediction Function](#2-define-sentiment-prediction-function)  
   - [Apply Sentiment Analysis](#3-apply-sentiment-analysis)  
   - [Filter Positive Responses](#4-filter-positive-responses)  
   - [Visualize Sentiment Distribution](#5-visualize-sentiment-distribution)  
   - [Sentiment Polarity Analysis](#6-sentiment-polarity-analysis)  

4. [Thematic Analysis Module](#thematic-analysis-module)  
   - [Preprocessing Text for LDA](#1-preprocessing-text-for-lda)  
   - [Train LDA Model](#2-train-lda-model)  
   - [Extract and Assign Themes](#3-extract-and-assign-themes)  
   - [Analyze Confidence Scores by Theme](#4-analyze-confidence-scores-by-theme)  
   - [Generate Word Clouds for Themes](#5-generate-word-clouds-for-themes)

5. [Region-Specific Analysis: California Feedback](#region-specific-analysis-california-feedback)  
   - [Filter California-Specific Responses](#filter-california-specific-responses)  
   - [Vectorize and Run LDA for California Feedback](#vectorize-and-run-lda-for-california-feedback)  
   - [Display California-Specific Themes](#display-california-specific-themes)  
   - [Sentiment Analysis for California-Specific Responses](#sentiment-analysis-for-california-specific-responses)

6. [Visualizations](#visualizations)  
   - [Sentiment Distribution (Donut Chart)](#visualization-sentiment-distribution-donut-chart)  
   - [Confidence Score Distribution by Sentiment Prediction](#visualization-confidence-score-distribution-by-sentiment-prediction)

## Project Overview
This project analyzes the **TPAS Survey Dataset** to extract meaningful insights and apply computational techniques, including machine learning and natural language processing (NLP). The code is designed to run in [Google Colab](https://colab.research.google.com), ensuring accessibility and ease of use.

## Features
- Data analysis and exploration using Python.
- Integration of NLP models using `transformers` and PyTorch.
- Survey data pre-processing and visualization.

## Getting Started

### Prerequisites
To run the notebook, you will need:
1. A Google account to access Google Colab.
2. A copy of the dataset (`TPAS survey dataset.xlsx`) uploaded to your Google Drive.

### How to Use
1. **Open the Notebook in Colab:**
   - Click [here](https://colab.research.google.com/) to open Google Colab.
   - Upload the notebook (`TPAS_SURVEY.ipynb`) to your Colab workspace.
   
2. **Connect to Google Drive:**
   - Mount your Google Drive to access the dataset by running the relevant cell in the notebook.

3. **Install Dependencies:**
   - Install required libraries using:
     ```python
     !pip install transformers torch
     ```

4. **Run the Notebook:**
   - Execute the cells sequentially to analyze the dataset.

## Dependencies
The following Python libraries are required:
- `pandas`
- `transformers`
- `torch`

Install all dependencies directly in Colab by running:
```python
!pip install transformers torch
  ```

## Sentiment Analysis Module

This module applies sentiment analysis to survey responses to determine if the sentiment is `POSITIVE` or `NEGATIVE`. It uses the `distilbert-base-uncased-finetuned-sst-2-english` model from Hugging Face.

### Code Overview

#### 1. Load Pre-trained Model

The sentiment analysis pipeline is loaded using Hugging Face's `pipeline` function:
```python
from transformers import pipeline
sentiment_pipeline = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
  ```
#### 2. Define Sentiment Prediction Function

A function `predict_sentiment` is defined to predict the sentiment of a given text. It ensures compatibility with the model by truncating text to 512 tokens:

```python
def predict_sentiment(text):
    try:
        result = sentiment_pipeline(text[:512])  # Truncate text to 512 tokens max for BERT-based models
        return result[0]['label']
    except Exception as e:
        return None
```
#### 3. Apply Sentiment Analysis

The function is applied to the `Responses` column of the dataset, creating a new column `Predicted_Sentiment`:

```python
df['Predicted_Sentiment'] = df['Responses'].apply(predict_sentiment)
```

#### 4. Filter Positive Responses

Optional: Filter and store responses with a `POSITIVE` sentiment:

```python
positive_responses = df[df["Predicted_Sentiment"] == 'POSITIVE']
```
#### 5. Visualize Sentiment Distribution

A sliced donut chart is created to visualize the distribution of positive and negative sentiments:

```python
import matplotlib.pyplot as plt

# Calculating the sentiment summary
sentiment_summary = df['Predicted_Sentiment'].value_counts().reset_index()
sentiment_summary.columns = ['Sentiment', 'Count']

# Data for the pie chart
labels = sentiment_summary['Sentiment']
sizes = sentiment_summary['Count']
colors = ['green', 'red']

# Creating the donut chart with a slice (exploded effect)
explode = (0.1, 0)  # Slightly "explode" the first slice (Positive)

# Plot the donut chart
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%', startangle=140, wedgeprops={'edgecolor': 'white'})

# Add a circle in the center to create the donut effect
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Title and formatting
plt.title('Distribution of Positive and Negative Sentiments (Sliced Donut Chart)')
plt.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle
plt.show()
```
#### 6. Sentiment Polarity Analysis

Sentiment polarity is calculated for each response using **TextBlob**. This identifies the most positive and most negative feedback based on sentiment polarity scores.

```python
from textblob import TextBlob

# Convert 'Responses' column to string type
df['Responses'] = df['Responses'].astype(str)

# Get sentiment polarity for each response
df['Sentiment_Polarity'] = df['Responses'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Sort by the most positive and negative polarity
most_positive = df[df['Sentiment_Polarity'] > 0].sort_values(by='Sentiment_Polarity', ascending=False).head(10)
most_negative = df[df['Sentiment_Polarity'] < 0].sort_values(by='Sentiment_Polarity').head(10)

# Display most positive and most negative responses
print("\nMost Positive Feedback:")
print(most_positive[['Responses', 'Sentiment_Polarity']])

print("\nMost Negative Feedback:")
print(most_negative[['Responses', 'Sentiment_Polarity']])
```

#### 7. Define Sentiment Prediction Function

The function `predict_sentiment` is updated to return both the sentiment label and confidence score:

```python
def predict_sentiment(text):
    try:
        result = sentiment_pipeline(text[:512])  # Truncate text to 512 tokens max for BERT-based models
        label = result[0]['label']
        score = result[0]['score']  # Confidence score for polarity
        return label, score
    except Exception as e:
        return None, None
```
### Thematic Analysis Module

This module applies thematic analysis to the negative responses in the dataset using **Latent Dirichlet Allocation (LDA)** for topic modeling.

#### Code Overview

##### 1. Preprocessing Text for LDA

Custom stop words are defined and combined with default English stop words to ensure the thematic analysis focuses on meaningful content:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Define custom stop words and combine with default English stop words
custom_stop_words = {'park', 'parking', 'make', 'area', 'know', 'issue', 'san', 'causes', 'truckers', 'available', 'los', 
                     'pull', 'don', 'cities', '10', 'way', 'country', 'big', 'able', 'want', 'states', 'state', 'lack', 
                     'dont', 'need', 'needs', 'pay', 'driving', 'stops', 'road', 'driver', 'drivers', 'drive', 'rest', 
                     'giving', 'use', 'would', 'like', 'get', 'also', 'just', 'still', 'time', 'place', 'trucks', 'truck', 
                     'park', 'parking', '_x000d_', 'areas', 'area', 'stop', 'doing', 'place', 'taking', 'right', 'going', 
                     'letting', 'places'}

# Combine custom stop words with default stop words
all_stop_words = list(ENGLISH_STOP_WORDS.union(custom_stop_words))

# Vectorize the text for LDA topic modeling
vectorizer = CountVectorizer(stop_words=all_stop_words, max_features=1000)
negative_texts = negative_responses['Responses'].tolist()
X_negative = vectorizer.fit_transform(negative_texts)
```
##### 2. Train LDA Model

An **LDA model** is trained to extract themes from the negative feedback, identifying key topics within the dataset:

```python
from sklearn.decomposition import LatentDirichletAllocation

# LDA model for topic extraction
num_topics = 5  # Number of topics to extract
lda = LatentDirichletAllocation(
    n_components=num_topics,
    random_state=42,
    doc_topic_prior=0.1,  # Alpha: Prior for document-topic distribution
    topic_word_prior=0.01  # Beta: Prior for topic-word distribution
)

# Train the LDA model on the vectorized negative responses
lda.fit(X_negative)
```
##### 3. Extract and Assign Themes

The top words for each topic are extracted for interpretation, and themes are assigned to each negative response.

###### Extract Themes

The top words for each topic are identified using the trained LDA model:

```python
# Extract themes
themes = {}
for idx, topic in enumerate(lda.components_):
    top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-10 - 1:-1]]  # Top 10 words
    themes[f"Theme {idx + 1}"] = top_words
```
###### LDA Components and Word Relevance

- **`lda.components_`**: Represents the word distributions across topics.
- **`topic.argsort()`**: Sorts words by their relevance to each topic.
- **`vectorizer.get_feature_names_out()`**: Retrieves the words corresponding to the vectorized indices.

This step helps interpret the themes by showing the most relevant words associated with each topic, enabling a better understanding of the key subjects present in the feedback.

Each response is analyzed to determine its dominant theme:
```python
# Assign themes to each response
def get_dominant_theme(text):
    transformed_text = vectorizer.transform([text])  # Transform text to match vectorizer format
    topic_distribution = lda.transform(transformed_text)  # Get topic distribution for the text
    dominant_topic = topic_distribution.argmax()  # Identify the dominant topic
    return f"Theme {dominant_topic + 1}"

# Assign themes to negative responses
negative_responses = negative_responses.copy()
negative_responses['Theme'] = negative_responses['Responses'].apply(get_dominant_theme)
```
- **`vectorizer.transform`**: Converts the text into the same format as used during training.
- **`lda.transform`**: Computes the topic distribution for the given text.
- **`argmax()`**: Identifies the topic with the highest probability.

- ##### 4. Analyze Confidence Scores by Theme

The average polarity score for each theme is calculated and visualized to provide insights into the sentiment confidence levels across different themes.

###### Calculate Average Confidence Score

The average polarity score for each theme is computed by grouping the data by theme:

```python
# Calculate average confidence score per theme
theme_confidence = negative_responses.groupby('Theme')['Polarity_Score'].mean().reset_index()
negative_responses['Theme'] = negative_responses['Responses'].apply(get_dominant_theme)
```
###### Explanation of Methods Used

- **`groupby('Theme')`**: Groups the responses by their assigned themes, allowing calculations to be performed within each theme category.
- **`mean()`**: Calculates the average polarity score for each theme, summarizing the sentiment intensity for that theme.
- **`reset_index()`**: Resets the index of the grouped data to prepare it for further processing or visualization.

These methods ensure the data is structured appropriately for analysis and visualization.

```python
import matplotlib.pyplot as plt

# Plot average confidence score by theme
plt.figure(figsize=(10, 6))
plt.barh(theme_confidence['Theme'], theme_confidence['Polarity_Score'], color='red')
plt.xlabel('Average Confidence Score')
plt.ylabel('Theme')
plt.title('Average Confidence Score by Theme in Negative Feedback')
plt.gca().invert_yaxis()  # Invert the y-axis for better readability
plt.show()
```
###### Explanation of Visualization Methods

- **`barh()`**: Creates a horizontal bar chart where themes are displayed on the y-axis and their corresponding confidence scores on the x-axis, providing a clear visual representation of the data.
- **`invert_yaxis()`**: Reverses the order of the themes on the y-axis, ensuring better readability and alignment with the descending order of scores.

These methods enhance the clarity and usability of the visualization, making it easier to interpret the average confidence scores for each theme.
### Key Outputs

- **Interpreted Themes with Top Words**: Displays the top words associated with each identified theme, offering insights into the core concepts represented by the themes.
- **Average Confidence Score by Theme**: A bar chart visualizing the distribution of confidence scores across themes, highlighting the intensity of sentiment for each theme.

This analysis provides thematic insights into negative feedback, enabling a deeper understanding of key areas for improvement based on the responses.

### Thematic Analysis Module

This module applies thematic analysis to the negative responses in the dataset using **Latent Dirichlet Allocation (LDA)** for topic modeling and word cloud generation.

#### Code Overview

##### 5. Generate Word Clouds for Themes

Word clouds are generated for each theme to visualize the most frequently occurring words within each category of negative feedback.

###### Assign Theme Labels

Labels are assigned to themes for better interpretability:

```python
from wordcloud import WordCloud

# Assign labels to each theme for clarity
theme_labels = {
    'Theme 1': 'General Safety and Accessibility Concerns',
    'Theme 2': 'Infrastructure and Facility Conditions Along Highways',
    'Theme 3': 'Limited Facility Availability and Operational Hours',
    'Theme 4': 'Cleanliness and Space Management Issues',
    'Theme 5': 'Speed Limits and Traffic Flow Challenges',
}

negative_responses['Theme_Labeled'] = negative_responses['Theme'].map(theme_labels)
```
###### Explanation of Key Steps

- **`theme_labels`**: A dictionary mapping each theme to a descriptive label, providing more context for each identified topic. This makes the themes more understandable and relevant to the analysis.
- **`map()`**: Maps the descriptive theme labels to the corresponding rows in the dataset, replacing the original theme codes with meaningful labels.

This step ensures that the themes are clearly labeled, enhancing interpretability and facilitating the generation of word clouds for each theme. It provides a structured and visually intuitive representation of the dataset's thematic content.

##### Generate Word Clouds

For each labeled theme, a word cloud is generated to highlight prominent terms associated with that theme. This visualization provides an intuitive understanding of the key concerns and issues reflected in the negative feedback.

###### Code for Generating Word Clouds

```python
# Generate word clouds for each theme
for theme, label in theme_labels.items():
    theme_text = " ".join(negative_responses[negative_responses['Theme_Labeled'] == label]['Responses'].tolist())

    # Create a word cloud
    wordcloud = WordCloud(
        width=800, height=400, background_color='white',
        stopwords=all_stop_words, colormap='Reds'
    ).generate(theme_text)

    # Plot the word cloud
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {label}')
    plt.show()
```

###### Explanation of Key Methods

- **`WordCloud()`**: Creates a word cloud object with specified parameters such as size, background color, and colormap.
- **`generate()`**: Generates the word cloud based on the input text, highlighting the most frequent words.
- **`imshow()`**: Displays the generated word cloud.

###### Key Outputs

- **Word Clouds for Each Theme**: Visually represents the most frequent terms associated with each theme, providing insights into the dominant concerns and recurring issues in the negative feedback.

This visualization tool enhances the interpretability of the thematic analysis by focusing on the most salient terms within each theme, making it easier to understand and address key issues.
### Thematic Analysis Module

This module applies thematic analysis to the negative responses in the dataset using **Latent Dirichlet Allocation (LDA)** for topic modeling and region-specific analysis.

#### Code Overview

##### Region-Specific Analysis: California Feedback

This section filters responses specifically mentioning California and performs topic modeling on the California-specific feedback.

###### Filter California-Specific Responses

Responses mentioning California are extracted for focused analysis:

```python
# Filter responses specifically mentioning California
california_responses = negative_responses[negative_responses['Responses'].str.contains("california|CA|California", case=False)]
```
- **`str.contains()`**: Identifies rows containing variations of "California" (case insensitive) to ensure all relevant responses are captured.

This filtering step narrows the dataset to feedback specifically mentioning California, enabling a focused thematic analysis for region-specific insights.
### Vectorize and Run LDA for California Feedback

The extracted California-specific feedback is vectorized, and an LDA model is trained to identify sub-themes within the feedback.

#### Code Overview

```python
# Vectorize the text for California-specific topic modeling
X_california = vectorizer.fit_transform(california_responses['Responses'])

# Run LDA on California-specific feedback
lda_california = LatentDirichletAllocation(
    n_components=3,  # Adjust based on expected sub-themes
    random_state=42,
    doc_topic_prior=0.1,  # Alpha
    topic_word_prior=0.01  # Beta
)
lda_california.fit(X_california)
```

#### Explanation of Key Parameters

- **`n_components`**: Specifies the number of sub-themes to extract. This parameter can be adjusted based on the data and the level of granularity required.
- **`doc_topic_prior (Alpha)`**: Controls the distribution of topics per document. Lower values result in fewer dominant topics per document, making each document more focused on specific themes.
- **`topic_word_prior (Beta)`**: Controls the distribution of words per topic. Lower values emphasize a smaller set of dominant words, leading to more distinct topics.

#### Display California-Specific Themes

The top words for each sub-theme are extracted and displayed for interpretation, helping to provide insights into the prominent topics within California-specific feedback.
#### Explanation of Key Methods

- **`lda.components_`**: Represents the word distributions across sub-themes, showing the importance of each word for every sub-theme.
- **`topic.argsort()`**: Sorts words by their relevance to each sub-theme, allowing the extraction of the most significant terms.
- **`get_feature_names_out()`**: Retrieves the terms corresponding to the vectorized indices, making the sub-themes interpretable.

#### Key Outputs

- **California-Specific Themes**: Displays the sub-themes extracted from California-specific feedback, including the top words that define each sub-theme.

This analysis provides targeted insights into feedback related to California, enabling more actionable regional strategies and highlighting the most critical issues raised in the responses.
### Sentiment Analysis for California-Specific Responses

This section performs sentiment analysis on California-specific responses using Hugging Face's sentiment analysis pipeline. Results include sentiment predictions, confidence scores, and visualizations to summarize the findings.

#### Code Overview

```python
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Hugging Face sentiment analysis pipeline
sentiment_pipeline = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

# Define a function to get sentiment prediction and confidence score using Hugging Face model
def predict_sentiment(text):
    result = sentiment_pipeline(text[:512])  # Truncate to 512 tokens for compatibility
    label = result[0]['label']
    score = result[0]['score']
    return label, score

# Filter for California-specific responses
california_responses = negative_responses[negative_responses['Responses'].str.contains("california|California|CA", case=False)]

# Calculate sentiment prediction and confidence score
california_responses[['Sentiment_Prediction', 'Confidence_Score']] = california_responses['Responses'].apply(lambda x: pd.Series(predict_sentiment(x)))

# Display results
print("California-specific Responses with Polarity, Sentiment Prediction, and Confidence Score:")
california_responses[['Responses', 'Sentiment_Prediction', 'Confidence_Score']].head()
```
### Visualization: Sentiment Distribution (Donut Chart)

```python
# Count of sentiment predictions (positive vs. negative)
sentiment_counts = california_responses['Sentiment_Prediction'].value_counts()

# Create a donut chart
plt.figure(figsize=(8, 8))
plt.pie(
    sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
    startangle=140, colors=['salmon', 'lightblue'], wedgeprops={'edgecolor': 'white'}
)
# Add a white circle in the center to create the donut effect
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Title and display
plt.title('Distribution of Positive and Negative Sentiments for California-specific Responses')
plt.show()
```
### Visualization: Confidence Score Distribution by Sentiment Prediction

```python
# Distribution of confidence scores by sentiment prediction
plt.figure(figsize=(10, 6))
sns.violinplot(data=california_responses, x='Sentiment_Prediction', y='Confidence_Score', hue='Sentiment_Prediction', dodge=False, legend=False)
plt.title('Confidence Score Distribution by Sentiment Prediction')
plt.xlabel('Sentiment Prediction')
plt.ylabel('Confidence Score')
plt.show()
```
#### Key Outputs

1. **California-Specific Responses**: Displays a table containing:
   - Responses mentioning California.
   - Sentiment predictions (`POSITIVE` or `NEGATIVE`).
   - Confidence scores indicating the certainty of the predictions.

2. **Sentiment Distribution**: A donut chart that visualizes the proportion of positive and negative sentiment predictions, providing a clear overview of sentiment polarity for California-specific feedback.

3. **Confidence Score Distribution**: A violin plot that illustrates the spread of confidence scores for each sentiment category, showcasing variations in the model's prediction confidence.

These visualizations and analyses provide detailed insights into the sentiment of California-specific feedback and highlight the model's certainty in its predictions.

