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
