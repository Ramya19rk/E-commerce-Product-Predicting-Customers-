# E-commerce-Product-Predicting-Customers-

Problem Statement:
"Develop a comprehensive data science project incorporating classification, image processing, natural language processing (NLP), and recommendation system tasks using E-commerce data, including exploratory data analysis, feature engineering, model building, and interactive visualization in Streamlit."

NAME : RAMYA KRISHNAN A

BATCH: DW75DW76

DOMAIN : DATA SCIENCE

DEMO VIDEO URL : 

Linked in URL : www.linkedin.com/in/ramyakrishnan19

# Libraries for Preprocessing

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    from sklearn.preprocessing import LabelEncoder
    import statistics as stats
    from scipy import stats
    import streamlit as st
    from streamlit_option_menu import option_menu

# Libraries for Image Processing

    import pytesseract
    import os
    import cv2
    import spacy
    from spacy import displacy

# Libraries for NLP analysis

    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.stem import PorterStemmer
    import geopandas as gpd
    import folium
    from streamlit_folium import folium_static

# Libraries for ML Process

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc , accuracy_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import pickle

# Libraries for Recommedation Process

    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.frequent_patterns import fpgrowth, association_rules
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    from sklearn.metrics.pairwise import cosine_similarity
    from itertools import chain

# Project Overview

# Domain and Technology

This project operates in the E-commerce domain, utilizing data-driven insights and machine learning models for customer prediction. The technology stack includes Python, Streamlit, and various machine learning libraries for classification and analysis tasks.

# Project Structure

# 1. Home

• Overview: Provides an introduction to the project, outlining the domain, technologies used, and an overview of the project's objectives.

# 2. Customer Prediction

## Form Structure:

• Input parameters: count hit, section count, device used.

• Users input data to predict customer conversion.

## Tabs:

### Extra Classifier Model:

• Column 1: Predict customer conversion.

• Column 2: Live prediction metrics (accuracy, precision, recall, F1 score).

### Random Forest Classifier Model:

• Column 1: Predict customer conversion.

• Column 2: Live prediction metrics (accuracy, precision, recall, F1 score).

### Decision Tree Classifier Model:

• Column 1: Predict customer conversion.

• Column 2: Live prediction metrics (accuracy, precision, recall, F1 score).

# 3. EDA Analysis

## Tabs:

### Correlation Matrix:

• Displays the pairwise correlation map of the given data.

### Histograms:

• Select box for column names to view histograms for specific columns.

### Relationship Plot:

• Scatter map with a select box for customer and non-customer, allowing users to visualize the distribution.

• Bar chart with radio buttons for specific relationship views.

• Pie chart with radio buttons for specific relationship views.

# 4. Upload

## Image Uploader:

• Allows users to upload images and view various processed versions (grayscale, blurred, edge detected, sharpened).

• Extracts words from images and displays the saved path.

# 5. NLP Analysis

## Text Processing:

• Users enter text for analysis.

• Outputs filtered tokens, stemming results, keywords, named entities, word count.

• Displays sentiment analysis in bar chart format.

# 6. Recommendation

## Product Recommendation:

• Users enter a product name to receive 10 product recommendations.

• If the entered product is new, random five products are suggested.

# Instructions

• Select a menu option from the sidebar to explore different functionalities.

• Follow the provided forms and inputs to interact with various features.

• Analyze live predictions, visualizations, and recommendations based on your inputs.










