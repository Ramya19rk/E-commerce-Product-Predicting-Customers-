import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import pickle
import sklearn
import pytesseract
import os
import cv2
import spacy
from spacy import displacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer
import geopandas as gpd
import folium
from streamlit_folium import folium_static
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Setting up page configuration

st.set_page_config(page_title= "E-Commerce Predicting Customers",
                   layout= "wide",
                   initial_sidebar_state= "expanded"                   
                  )

# Creating Background

def setting_bg():
    st.markdown(f""" <style>.stApp {{
                    background:url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQusP3jn05fepU72kFxDCVTboLA6_tlJdZIxw&usqp=CAU");
                    background-size: cover}}
                </style>""", unsafe_allow_html=True)
setting_bg()

# Creating option menu in the side bar

with st.sidebar:

    selected = option_menu("Menu", ["Home","Customers","EDA Analysis","Upload","NLP Analysis","Recommend"], 
                           icons=["house","list-task","bar-chart-line","cloud-upload","pencil-square","star"],
                           menu_icon= "menu-button-wide",
                           default_index=0,
                           styles={"nav-link": {"font-size": "20px", "text-align": "left", "margin": "-2px", "--hover-color": "blue"},
                                   "nav-link-selected": {"background-color": "blue"}}
                          )
    
# Home Menu

if selected == 'Home':

    st.title(":green[*E-COMMERCE PRODUCT [PREDICTING CUSTOMERS]*]")
    
    col1, col2 = st.columns(2)

    with col1:
        
        col1.markdown("# ")
        col1.markdown("# ")
        col1.markdown("## :violet[*Domain*] : E_Commerce")
        col1.markdown("# ")
        col1.markdown("## :violet[*Technologies used*] : Python, Pandas, Numpy, Matplotlib, Seaborn, Scikit-learn, OCR, Nltk, Streamlit.")
        col1.markdown("# ")
        col1.markdown("## :violet[*Overview*] : ")
        col1.markdown("## :blue[*Predicting Customers*] : Build Classification Model to Predict Customers")
        col1.markdown("# ")
        col1.markdown("## :blue[*Image Processing*] : Utilize image processing techniques for enhanced OCR accuracy.")
        col1.markdown("# ")
        col1.markdown("## :blue[*NLP Analysis*] :  Build NLP model to get better insights.")
        col1.markdown("# ")
        col1.markdown("## :blue[*Recommendation*] :  Build Recommendation model to get the best products.")
        col1.markdown("# ")
        col1.markdown("## :blue[*EDA Analysis*] :  Build EDA model to get visualizations on the data.")

    with col2:
        col2.markdown("# ")
        col2.markdown("# ")
        col2.image("/Users/arul/Downloads/ecomm2.jpg")
        col2.markdown("# ")
        col2.image("/Users/arul/Downloads/ecom.jpg")
        col2.markdown("# ")
        col2.image("/Users/arul/Downloads/ecom3.jpg")

if selected == 'Customers':
    tab1,tab2,tab3 = st.tabs(["$\huge  EXTRATREESCLASSIFIER $", "$\huge RANDOMFORESTCLASSIFIER $", "$\huge DECISIONTREECLASSIFIER $"])

    # EXTRATREECLASSIFIER

    with tab1:
        col1,col2 = st.columns(2)

        with col1:
            col1.markdown("## :blue[*EXTRATREESCLASSIFIER*]")

            with st.form("form1"):

                count_hit = st.number_input('CLICK COUNT')

                count_session = st.number_input('SESSION COUNT')

                device_categories = {0: 'Desktop', 1: 'Mobile', 2: 'Tablet'}

                device_deviceCategory = st.selectbox('DEVICE CATEGORY', options=device_categories.keys(), format_func=lambda x: device_categories[x])

                submit_button = st.form_submit_button(label="PREDICT CUSTOMER")

                if submit_button is not None:
                    with open(r"model4.pkl", 'rb') as f:
                        loaded_model = pickle.load(f)

                        new_sample = np.array(
                    [[count_hit, count_session, device_deviceCategory]])                       
                        new_pred = loaded_model.predict(new_sample)[0]
                        
                        if new_pred == 1.0:                       
                            st.markdown(
                                "<h1 style='font-size: 40px;'><span style='color: green;'>Convert Has Customer </span></h1>",
                                unsafe_allow_html=True)

                        elif new_pred == 0.0:
                            #print(x)
                            st.markdown(
                                "<h1 style='font-size: 40px;'><span style='color: red;'>Not Convert Has Customer</span> </h1>",
                                unsafe_allow_html=True)

        with col2:
            
            dd = pd.read_csv('predicting.csv')

            x = dd.drop('has_converted',axis=1)
            y = dd['has_converted']
            def calculate_metrics(model, xtest, ytest):
                predicted_values = model.predict(xtest)
                acc = metrics.accuracy_score(ytest, predicted_values)
                prec = metrics.precision_score(ytest, predicted_values)
                recall = metrics.recall_score(ytest, predicted_values)
                f1 = metrics.f1_score(ytest, predicted_values)
                return acc, prec, recall, f1

            # Split the data into training and testing sets
            xtrain,xtest,ytrain,ytest = train_test_split(x.values,y,test_size=0.2)

            # Load the pickled model
            with open(r"model4.pkl", 'rb') as f:
                loaded_model = pickle.load(f)

            # Calculate metrics
            acc, prec, recall, f1 = calculate_metrics(loaded_model, xtest, ytest)

            # Display in Streamlit
            col2.markdown("## :blue[*MODEL EVALUATION*]")
            

            # Display additional metrics
            st.markdown(
                f'<h1 style="color: violet; display: inline;">Accuracy:</h1>'
                f'<h1 style="color: brown; display: inline;"> {acc}</h1>',
                unsafe_allow_html=True)
            st.markdown(
                f'<h1 style="color: violet; display: inline;">Precision:</h1>'
                f'<h1 style="color: brown; display: inline;"> {prec}</h1>',
                unsafe_allow_html=True)
            st.markdown(
                f'<h1 style="color: violet; display: inline;">Recall:</h1>'
                f'<h1 style="color: brown; display: inline;"> {recall}</h1>',
                unsafe_allow_html=True)
            st.markdown(
                f'<h1 style="color: violet; display: inline;">F1 Score:</h1>'
                f'<h1 style="color: brown; display: inline;"> {f1}</h1>',
                unsafe_allow_html=True)
        
    with tab2:
        col1,col2 = st.columns(2)

        with col1:
            col1.markdown("## :blue[*RANDOMFORESTCLASSIFIER*]")

            with st.form("form2"):

                count_hit = st.number_input('CLICK COUNT')

                count_session = st.number_input('SESSION COUNT')

                device_categories = {0: 'Desktop', 1: 'Mobile', 2: 'Tablet'}

                device_deviceCategory = st.selectbox('DEVICE CATEGORY', options=device_categories.keys(), format_func=lambda x: device_categories[x])


                submit_button = st.form_submit_button(label="PREDICT CUSTOMER")

                if submit_button is not None:
                    with open(r"model6.pkl", 'rb') as f:
                        loaded_model = pickle.load(f)

                        new_sample = np.array(
                    [[count_hit, count_session, device_deviceCategory]])                       
                        new_pred = loaded_model.predict(new_sample)[0]
                        
                        if new_pred == 1.0:                       
                            st.markdown(
                                "<h1 style='font-size: 40px;'><span style='color: green;'>Convert Has Customer</span> </h1>",
                                unsafe_allow_html=True)

                        elif new_pred == 0.0:
                            #print(x)
                            st.markdown(
                                "<h1 style='font-size: 40px;'><span style='color: red;'>Not Convert Has Customer</span> </h1>",
                                unsafe_allow_html=True)

        with col2:

            dd = pd.read_csv('predicting.csv')

            x = dd.drop('has_converted',axis=1)
            y = dd['has_converted']
            def calculate_metrics(model, xtest, ytest):
                predicted_values = model.predict(xtest)
                acc = metrics.accuracy_score(ytest, predicted_values)
                prec = metrics.precision_score(ytest, predicted_values)
                recall = metrics.recall_score(ytest, predicted_values)
                f1 = metrics.f1_score(ytest, predicted_values)
                return acc, prec, recall, f1

            # Split the data into training and testing sets
            xtrain,xtest,ytrain,ytest = train_test_split(x.values,y,test_size=0.2)

            # Load the pickled model
            with open(r"model6.pkl", 'rb') as f:
                loaded_model = pickle.load(f)
            # Calculate metrics
            acc, prec, recall, f1 = calculate_metrics(loaded_model, xtest, ytest)

            # Display in Streamlit
            col2.markdown("## :blue[*MODEL EVALUATION*]")
            

            # Display additional metrics
            st.markdown(
                f'<h1 style="color: violet; display: inline;">Accuracy:</h1>'
                f'<h1 style="color: brown; display: inline;"> {acc}</h1>',
                unsafe_allow_html=True)
            st.markdown(
                f'<h1 style="color: violet; display: inline;">Precision:</h1>'
                f'<h1 style="color: brown; display: inline;"> {prec}</h1>',
                unsafe_allow_html=True)
            st.markdown(
                f'<h1 style="color: violet; display: inline;">Recall:</h1>'
                f'<h1 style="color: brown; display: inline;"> {recall}</h1>',
                unsafe_allow_html=True)
            st.markdown(
                f'<h1 style="color: violet; display: inline;">F1 Score:</h1>'
                f'<h1 style="color: brown; display: inline;"> {f1}</h1>',
                unsafe_allow_html=True)
        

    with tab3:
        col1,col2 = st.columns(2)

        with col1:
            col1.markdown("## :blue[*DECISIONTREECLASSIFIER*]")

            with st.form("form3"):

                count_hit = st.number_input('CLICK COUNT')

                count_session = st.number_input('SESSION COUNT')

                device_categories = {0: 'Desktop', 1: 'Mobile', 2: 'Tablet'}

                device_deviceCategory = st.selectbox('DEVICE CATEGORY', options=device_categories.keys(), format_func=lambda x: device_categories[x])

                submit_button = st.form_submit_button(label="PREDICT CUSTOMER")

                if submit_button is not None:
                    with open(r"model7.pkl", 'rb') as f:
                        loaded_model = pickle.load(f)

                        new_sample = np.array(
                    [[count_hit, count_session, device_deviceCategory]])                       
                        new_pred = loaded_model.predict(new_sample)[0]
                        
                        if new_pred == 1.0:                       
                            st.markdown(
                                "<h1 style='font-size: 40px;'><span style='color: green;'>Convert Has Customer </span> </h1>",
                                unsafe_allow_html=True)

                        elif new_pred == 0.0:
                            #print(x)
                            st.markdown(
                                "<h1 style='font-size: 40px;'><span style='color: red;'>Not Convert Has Customer </span> </h1>",
                                unsafe_allow_html=True)

        with col2:

            dd = pd.read_csv('predicting.csv')

            x = dd.drop('has_converted',axis=1)
            y = dd['has_converted']
            def calculate_metrics(model, xtest, ytest):
                predicted_values = model.predict(xtest)
                acc = metrics.accuracy_score(ytest, predicted_values)
                prec = metrics.precision_score(ytest, predicted_values)
                recall = metrics.recall_score(ytest, predicted_values)
                f1 = metrics.f1_score(ytest, predicted_values)
                return acc, prec, recall, f1

            # Split the data into training and testing sets
            xtrain,xtest,ytrain,ytest = train_test_split(x.values,y,test_size=0.2)

            # Load the pickled model
            with open(r"model7.pkl", 'rb') as f:
                loaded_model = pickle.load(f)
            # Calculate metrics
            acc, prec, recall, f1 = calculate_metrics(loaded_model, xtest, ytest)

           # Display in Streamlit
            col2.markdown("## :blue[*MODEL EVALUATION*]")
            
            # Display additional metrics
            st.markdown(
                f'<h1 style="color: violet; display: inline;">Accuracy:</h1>'
                f'<h1 style="color: brown; display: inline;"> {acc}</h1>',
                unsafe_allow_html=True)
            st.markdown(
                f'<h1 style="color: violet; display: inline;">Precision:</h1>'
                f'<h1 style="color: brown; display: inline;"> {prec}</h1>',
                unsafe_allow_html=True)
            st.markdown(
                f'<h1 style="color: violet; display: inline;">Recall:</h1>'
                f'<h1 style="color: brown; display: inline;"> {recall}</h1>',
                unsafe_allow_html=True)
            st.markdown(
                f'<h1 style="color: violet; display: inline;">F1 Score:</h1>'
                f'<h1 style="color: brown; display: inline;"> {f1}</h1>',
                unsafe_allow_html=True)
        
if selected == 'Upload':
    # Function to perform image pre-processing
    def preprocess_image(uploaded_file):
        # Convert the uploaded file to a numpy array (image data)
        image_data = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Apply edge detection using Canny
        edges_image = cv2.Canny(blurred_image, 50, 150)

        return image_data, gray_image, blurred_image, edges_image

    # Function to sharpen the image
    def sharpen_image(image):
        # Create a sharpening filter kernel
        kernel = np.array([[-1, -1, -1],
                        [-1,  9, -1],
                        [-1, -1, -1]])

        # Apply the sharpening kernel
        sharpened_image = cv2.filter2D(image, -1, kernel)
        return sharpened_image

    # Function to perform OCR on the pre-processed image
    def perform_ocr(image):
        # Use Tesseract OCR to extract text
        text_from_image = pytesseract.image_to_string(image)
        return text_from_image

    # Streamlit App
    def main():
        st.title(":red[Image Processing and OCR Demo]")

        # File uploader for image selection
        uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Display the uploaded image
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

            # Perform image pre-processing
            original, gray, blurred, edges = preprocess_image(uploaded_file)

            # Display images in columns
            col1, col2, col3, col4 = st.columns(4)
            col1.image(gray, caption="Grayscale Image", use_column_width=True)
            col2.image(blurred, caption="Blurred Image", use_column_width=True)
            col3.image(edges, caption="Edge Detected Image", use_column_width=True)

            # Sharpen the original image
            sharpened = sharpen_image(original)
            col4.image(sharpened, caption="Sharpened Image", use_column_width=True)

            # Perform OCR on the edge-detected image
            ocr_result = perform_ocr(edges)

            # Display OCR result
            st.subheader(":blue[Extracted Text:]")
            st.write(ocr_result)

            # Create the full path for the uploaded image
            folder_name = "uploaded_images"
            file_name = uploaded_file.name  # Use the name of the uploaded file
            image_path = os.path.join(os.path.expanduser("~"), "Documents", folder_name, file_name)
            
            # Display the created image path
            st.subheader(":blue[Image Path:]")
            st.write(image_path)

    if __name__ == "__main__":
        main()

if selected == 'NLP Analysis':
   
  # Load spaCy English model
    nlp = spacy.load('en_core_web_sm')

    # Load NLTK Porter Stemmer
    ps = PorterStemmer()

    # NLP pre-processing with stemming
    def preprocess_text(text):
        # Tokenization
        tokens = word_tokenize(text)

        # Remove stopwords and apply stemming
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [ps.stem(word.lower()) for word in tokens if word.isalnum() and word.lower() not in stop_words]

        return filtered_tokens

    # Keyword extraction
    def extract_keywords(text):
        doc = nlp(text)
        keywords = [token.text for token in doc if token.is_alpha and not token.is_stop]
        return keywords

    # Named Entity Recognition (NER)
    def ner_analysis(text):
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities

    # Sentiment analysis
    def sentiment_analysis(input_text):
        # Initialize SentimentIntensityAnalyzer
        sid = SentimentIntensityAnalyzer()

        # Get sentiment scores
        sentiment_scores = sid.polarity_scores(input_text)

        # Plotting Bar Chart for Sentiment Analysis
        sentiments = ['Negative', 'Neutral', 'Positive']
        sentiment_scores_list = [sentiment_scores[key] for key in ['neg', 'neu', 'pos']]
        colors = ['red', 'gray', 'green']

        fig, ax = plt.subplots()
        bars = ax.bar(sentiments, sentiment_scores_list, color=colors)
        ax.set_title('Sentiment Analysis')

        # Add labels on top of the bars
        for bar, score in zip(bars, sentiment_scores_list):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{score:.2f}', ha='center', va='bottom')

        # Display the plot using Streamlit
        st.pyplot(fig)
        

    # Streamlit App
    def main():
        st.title(":red[NLP Processing and Analysis]")

        # User input for text
        input_text = st.text_area("Enter the text:")

        if st.button("Process and Analyze"):
            # NLP pre-processing with stemming
            filtered_tokens = preprocess_text(input_text)

            # Stemming Output
            stemmed_output = [ps.stem(word) for word in filtered_tokens]

            # Keyword extraction
            keywords = extract_keywords(input_text)

            # Named Entity Recognition (NER)
            entities = ner_analysis(input_text)

            # Word Count
            word_count = len(word_tokenize(input_text))

            # Display in columns
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.subheader(":blue[Filtered Tokens:]")
                st.write(filtered_tokens)

            with col2:
                st.subheader(":blue[Stemming Output:]")
                st.write(stemmed_output)

            with col3:
                st.subheader(":blue[Keywords:]")
                st.write(keywords)

            with col4:
                st.subheader(":blue[Named Entities:]")
                st.write(entities)

            with col5:
                st.subheader(":blue[Word Count:]")
                st.markdown(f"<p style='font-size:24px;'>{word_count}</p>", unsafe_allow_html=True)

            # Sentiment analysis
            st.title(':violet[Sentiment Analysis]')
            sentiment_analysis(input_text)

    if __name__ == "__main__":
        main()

if selected == 'Recommend':
    st.title(":red[Recommendation Page]")

    def get_content_based_recommendations(product_name, df, cosine_similarities):
        matching_products = df[df['Products'] == product_name]

        if matching_products.empty:
            # Handle the case where the product is not found
            # Provide recommendations for random products
            random_products = df.sample(n=5)['Products'].tolist()
            return random_products

        product_index = matching_products.index[0]
        similar_products = list(enumerate(cosine_similarities[product_index]))

        # Sort the products based on similarity scores
        similar_products = sorted(similar_products, key=lambda x: x[1], reverse=True)

        # Exclude the input product from recommendations
        similar_products = [index for index, score in similar_products if index != product_index]

        # Filter out duplicate products (case-insensitive)
        unique_recommendations = set()
        filtered_recommendations = []
        for index in similar_products:
            product = df['Products'].iloc[index].lower()  # Convert to lowercase
            if product not in unique_recommendations:
                filtered_recommendations.append(df['Products'].iloc[index])
                unique_recommendations.add(product)

            # Break the loop if we have enough recommendations
            if len(filtered_recommendations) >= 10:
                break

        return filtered_recommendations[:10]

    def main():
        product_name_to_recommend = st.text_input("Enter a product name:")

        # Button to trigger recommendations
        if st.button("Get Recommendations"):
            if product_name_to_recommend:
                # Get content-based recommendations for the entered product
                content_based_recommendations = get_content_based_recommendations(
                    product_name_to_recommend, df_flat, loaded_cosine_similarities
                )

                # Display recommendations
                st.markdown(
                    f'<h1 style="color: violet;">Recommended Products for</h1>'
                    f'<h1 style="color: pink;">{product_name_to_recommend}:</h1>',
                    unsafe_allow_html=True)
            if not content_based_recommendations:
                st.markdown("No recommendations found. Here are some random products:")
                content_based_recommendations = get_content_based_recommendations(
                        "random_product", df_flat, loaded_cosine_similarities
                    )
            
            for recommendation in content_based_recommendations:
                    st.write(recommendation)
                    
    # Load the model and DataFrame
    with open('model_1.pkl', 'rb') as file:
        loaded_tfidf_vectorizer, loaded_cosine_similarities = pickle.load(file)

    df_3 = pd.read_pickle('data_recomend.pkl')

    # Flatten the product lists into individual rows
    df_flat = df_3.explode('Products')

    # Convert 0.0 values in Rating to 1 (purchased)
    df_flat['Rating'] = df_flat['Rating'].replace(0.0, 1)

    # Run the app
    if __name__ == "__main__":
        main()

if selected == 'EDA Analysis':

    st.title(":red[EDA Analysis Page]")
    
    df_1 = pd.read_csv('EDA_analysis.csv')

    tab1,tab2,tab3 = st.tabs(["$\huge  CORRELATION MATRIX $", "$\huge HISTOGRAM $", "$\huge RELATIONSHIP PLOT $"])

    with tab1:

        # Display the heatmap using Streamlit
        st.title(':violet[Pairwise Correlation Heatmap]')
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(df_1.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)

    with tab2:
        st.title(':violet[Univariate Analysis: Histograms for Each Feature]')

       # Dropdown for selecting feature
        selected_feature = st.selectbox('Select a feature', df_1.columns)

        # Display histogram for the selected feature
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(df_1[selected_feature], bins=30, edgecolor='black')
        ax.set_title(f'Histogram for {selected_feature}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')

        # Adjust layout
        fig.tight_layout()
        st.pyplot(fig)

    with tab3:

# Scatter Map:
                
        st.title(':blue[Visitor Conversion Map]')
        df = pd.read_csv('classification_data.csv')

        # Dropdown for selecting conversion status
        conversion_status = st.selectbox('Select Conversion Status', ['Converted', 'Not Converted'])

        # Filter DataFrame based on conversion status
        if conversion_status == 'Converted':
            df_filtered = df[df['has_converted'] == 1.0]
        else:
            df_filtered = df[df['has_converted'] == 0.0]

        # Plotly scatter map
        fig = px.scatter_mapbox(
            df_filtered,
            lat='geoNetwork_latitude',
            lon='geoNetwork_longitude',
            hover_name='geoNetwork_region',
            mapbox_style='carto-positron'
        )

        # Update layout
        fig.update_layout(
            mapbox_style='carto-positron',
            margin=dict(l=0, r=0, t=0, b=0)
        )

        # Display the map
        st.plotly_chart(fig)

# Bar Chart:

        st.title(':blue[Bar Chart for Selected Columns]')

        # Select box for choosing column
        
        selected_columns = st.radio('Select a column', ['device_deviceCategory', 'avg_visit_time', 'days_since_first_visit','single_page_rate'])

        if selected_columns == 'device_deviceCategory':
            fig = px.bar(df,
                                    title='Device Category',
                                    x="earliest_visit_number",
                                    y="device_deviceCategory",
                                    orientation='h',
                                    color='device_deviceCategory',
                                    color_continuous_scale=px.colors.sequential.Agsunset)
            st.plotly_chart(fig,use_container_width=True)

        
        if selected_columns == 'avg_visit_time':
            fig = px.bar(df,
                                    title='Average Visit Time',
                                    x="earliest_visit_number",
                                    y="avg_visit_time",
                                    orientation='h',
                                    color='avg_visit_time',
                                    color_continuous_scale=px.colors.sequential.Agsunset)
            st.plotly_chart(fig,use_container_width=True)
        
        if selected_columns == 'days_since_first_visit':
            fig = px.bar(df,
                                    title='Average Visit Time',
                                    x="earliest_visit_number",
                                    y="days_since_first_visit",
                                    orientation='h',
                                    color='days_since_first_visit',
                                    color_continuous_scale=px.colors.sequential.Agsunset)
            st.plotly_chart(fig,use_container_width=True)

        if selected_columns == 'single_page_rate':
            fig = px.bar(df,
                                    title='Rating Based on Channel',
                                    x="single_page_rate",
                                    y="channelGrouping",
                                    orientation='h',
                                    color='channelGrouping',
                                    color_continuous_scale=px.colors.sequential.Agsunset)
            st.plotly_chart(fig,use_container_width=True)

# Pie Chart
            
        st.title(':blue[Pie Chart for Selected Columns]')

        # Select box for choosing column

        selected_column = st.radio('Select a column', ['device_operatingSystem', 'earliest_medium','device_isMobile'])
       
        if selected_column == 'device_operatingSystem':

            fig = px.pie(df, values = 'time_on_site',
                        names = 'device_operatingSystem',
                        color_discrete_sequence=px.colors.cyclical.Edge,
                        hover_data=['device_operatingSystem'],
                        labels={'device_operatingSystem':'device_operatingSystem'}
                        )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

        if selected_column == 'earliest_medium':

            fig = px.pie(df, values = 'num_interactions',
                        names = 'earliest_medium',
                        color_discrete_sequence=px.colors.cyclical.Edge,
                        hover_data=['earliest_medium'],
                        labels={'medium_used':'earliest_medium'}
                        )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

        if selected_column == 'device_isMobile':

            fig = px.pie(df, values = 'avg_session_time',
                        names = 'device_isMobile',
                        color_discrete_sequence=px.colors.cyclical.Edge,
                        hover_data=['device_isMobile'],
                        labels={'Mobile_Users':'device_isMobile'}
                        )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
