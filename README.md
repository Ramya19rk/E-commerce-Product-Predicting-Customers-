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

<img width="1440" alt="Screenshot 2024-01-26 at 12 01 45 PM" src="https://github.com/Ramya19rk/E-commerce-Product-Predicting-Customers-/assets/145639838/c9996720-903e-4129-aa8c-2c53470beb94">


# 2. Customer Prediction

## Form Structure:

• Input parameters: count hit, section count, device used.

• Users input data to predict customer conversion.

## Tabs:

### Extra Classifier Model:

• Column 1: Predict customer conversion.

• Column 2: Live prediction metrics (accuracy, precision, recall, F1 score).

<img width="1440" alt="Screenshot 2024-01-26 at 12 01 55 PM" src="https://github.com/Ramya19rk/E-commerce-Product-Predicting-Customers-/assets/145639838/f993b350-2036-43fc-ae2d-c5f2ac401262">

### Random Forest Classifier Model:

• Column 1: Predict customer conversion.

• Column 2: Live prediction metrics (accuracy, precision, recall, F1 score).

<img width="1440" alt="Screenshot 2024-01-26 at 12 02 04 PM" src="https://github.com/Ramya19rk/E-commerce-Product-Predicting-Customers-/assets/145639838/4d1a0bd6-35a0-413b-8d45-6af22009aff7">


### Decision Tree Classifier Model:

• Column 1: Predict customer conversion.

• Column 2: Live prediction metrics (accuracy, precision, recall, F1 score).

<img width="1437" alt="Screenshot 2024-01-26 at 12 02 12 PM" src="https://github.com/Ramya19rk/E-commerce-Product-Predicting-Customers-/assets/145639838/4cc8ba2c-779c-4ffe-8d1e-99253eb23dcf">


# 3. EDA Analysis

## Tabs:

### Correlation Matrix:

• Displays the pairwise correlation map of the given data.

<img width="1440" alt="Screenshot 2024-01-26 at 12 02 23 PM" src="https://github.com/Ramya19rk/E-commerce-Product-Predicting-Customers-/assets/145639838/da9b5d42-561e-44cd-8e6c-99ba2efd601d">

### Histograms:

• Select box for column names to view histograms for specific columns.

<img width="1440" alt="Screenshot 2024-01-26 at 12 02 34 PM" src="https://github.com/Ramya19rk/E-commerce-Product-Predicting-Customers-/assets/145639838/70a83f58-9e7d-494b-9aa8-74512f0e3fb1">

### Relationship Plot:

• Scatter map with a select box for customer and non-customer, allowing users to visualize the distribution.

• Bar chart with radio buttons for specific relationship views.

• Pie chart with radio buttons for specific relationship views.

<img width="1440" alt="Screenshot 2024-01-26 at 12 02 47 PM" src="https://github.com/Ramya19rk/E-commerce-Product-Predicting-Customers-/assets/145639838/dd5743df-75f7-47a7-b620-5cad774ffa4a">

<img width="1439" alt="Screenshot 2024-01-26 at 12 02 54 PM" src="https://github.com/Ramya19rk/E-commerce-Product-Predicting-Customers-/assets/145639838/dcf1f76d-2195-4484-a9fb-26221f674bc6">

<img width="1440" alt="Screenshot 2024-01-26 at 12 03 09 PM" src="https://github.com/Ramya19rk/E-commerce-Product-Predicting-Customers-/assets/145639838/72914114-2be8-4481-b066-6d3e456d6d07">

# 4. Upload

## Image Uploader:

• Allows users to upload images and view various processed versions (grayscale, blurred, edge detected, sharpened).

• Extracts words from images and displays the saved path.

<img width="1434" alt="Screenshot 2024-01-26 at 12 03 17 PM" src="https://github.com/Ramya19rk/E-commerce-Product-Predicting-Customers-/assets/145639838/13e1bb01-cb3e-421c-988b-095b8ddf15b6">

# 5. NLP Analysis

## Text Processing:

• Users enter text for analysis.

• Outputs filtered tokens, stemming results, keywords, named entities, word count.

• Displays sentiment analysis in bar chart format.

<img width="1440" alt="Screenshot 2024-01-26 at 12 03 24 PM" src="https://github.com/Ramya19rk/E-commerce-Product-Predicting-Customers-/assets/145639838/dfcfa49d-bb4d-4d12-b29c-b251d809abad">

# 6. Recommendation

## Product Recommendation:

• Users enter a product name to receive 10 product recommendations.

• If the entered product is new, random five products are suggested.

<img width="1440" alt="Screenshot 2024-01-26 at 12 03 32 PM" src="https://github.com/Ramya19rk/E-commerce-Product-Predicting-Customers-/assets/145639838/8e652213-5400-4eb5-a911-c7e644119dcc">

# Instructions

• Select a menu option from the sidebar to explore different functionalities.

• Follow the provided forms and inputs to interact with various features.

• Analyze live predictions, visualizations, and recommendations based on your inputs.










