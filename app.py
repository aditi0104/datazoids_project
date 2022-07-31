import streamlit as st
import os
from PIL import Image
import numpy as np
import pandas as pd
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

file_upload_path = "/Users/madhavacherukuri/Downloads/Datazoids/uploads"
imagesPath = "/Users/madhavacherukuri/Downloads/Datazoids/images"

# creating a ResNet model which would return the features of an given image.
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# the uploaded file should be saved in some path for later usage
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join(file_upload_path, uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    image_array = image.img_to_array(img)
    expanded_image_array = np.expand_dims(image_array, axis=0)
    preprocessed_image = preprocess_input(expanded_image_array)
    result = model.predict(preprocessed_image).flatten()
    normalized_result = result / norm(result)
    
    return normalized_result

def recommend_images(features, features_list):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='auto', metric='cosine')
    neighbors.fit(features_list)

    distances, indices = neighbors.kneighbors([features], n_neighbors=10)

    return indices


# steps        
st.set_page_config(layout="wide", initial_sidebar_state='expanded')    
st.title('H&M Fashion Recommender System') 
sidebar_header = '''A demo to illustrate a recommender system that finds similar items to a given clothing article or recommend items for a customer using colloborative filtering method:'''

page_options = ["Find similar items",
                "Customer Recommendations"]

st.sidebar.info(sidebar_header)

page_selection = st.sidebar.radio("Try", page_options)
# articles_df = pd.read_csv('articles.csv')

models = ['Similar items based on image embeddings', 
          'Similar items based on text embeddings', 
          'Similar items based discriptive features', 
          'Similar items based on embeddings from TensorFlow Recommendrs model',
          'Similar items based on a combination of all embeddings']

#########################################################################################
#########################################################################################

if page_selection == "Find similar items":
    
    features_list = np.array(pickle.load(open('Downloads/Datazoids/embeddings.pkl','rb')))
    filenames = pickle.load(open('Downloads/Datazoids/filenames.pkl','rb'))
    uploaded_file_path = ''
    
    get_item = st.sidebar.button('Get Random Item') # button for random item selection
    if get_item:
        file_name = np.random.choice(filenames)
        uploaded_file_path = os.path.join(imagesPath, file_name)
        st.sidebar.text('Selected Item - ')
        st.sidebar.image(uploaded_file_path, width=200)
    else:
        # giving the user chance to upload image to view the related products in H&M.
        file = st.file_uploader("Select an image")
        if file is not None:
            if save_uploaded_file(file):
                uploaded_file_path = os.path.join(file_upload_path, file.name)
                
                st.text('Uploaded Image - ')
                display_image = Image.open(file)
                st.image(display_image, width=500)
            else:
                st.header("Some error occured in the file upload")
       
    if uploaded_file_path:
        # extracting features from the given image
        features = feature_extraction(uploaded_file_path, model)

        # get recommendention images
        indices = recommend_images(features, features_list)

        st.text("Related products are - ")

        # display the 5 boxes for the five images
        col1,col2,col3,col4,col5 = st.columns(5)

        with col1:
            st.image(os.path.join(imagesPath, filenames[indices[0][0]]))
        with col2:
            st.image(os.path.join(imagesPath, filenames[indices[0][1]]))
        with col3:
            st.image(os.path.join(imagesPath, filenames[indices[0][2]]))
        with col4:
            st.image(os.path.join(imagesPath, filenames[indices[0][3]]))
        with col5:
            st.image(os.path.join(imagesPath, filenames[indices[0][4]]))


#########################################################################################
#########################################################################################
if page_selection == "Customer Recommendations":

    customers_data = pickle.load(open('Downloads/Datazoids/customer_transactions_embeddings.pkl','rb'))
    implicit_als_embeddings = pd.read_csv('Downloads/Datazoids/submissions-Implicit-ALS.csv')
    customer_id = ''
    get_item = st.sidebar.button('Get Random Customer') # button for random user selection
    
    if get_item:
        customer_id = np.random.choice(customers_data['customer_id'])
    else:
        customer_id = st.text_input('Customer ID')
        st.write('The current user id is', customer_id)
        
    if customer_id:
        if customer_id in np.array(customers_data['customer_id']):
            st.sidebar.write('#### Customer history')
            
            customer_data = customers_data[customers_data['customer_id'] == customer_id]
            articles = np.array(customer_data['article_id'])[0]

            # Displaying the images of items that user have bought in the past.
            rows = [articles[i:i+3] for i in range(0, len(articles), 3)]
            for row in rows:
                with st.sidebar.container():
                    columns = st.columns(3)
                    for item, col in zip(row, columns):
                        col.image(os.path.join(imagesPath, str(item)[0:3], f'{str(item)}.jpg'), 100)

            predictions = np.array(implicit_als_embeddings[implicit_als_embeddings['customer_id'] == customer_id]['prediction'])[0].split(' ')

            st.write('The recommendations based on the users current purchase are -')
            
            # display the 5 boxes for the five images
            col1,col2,col3,col4,col5 = st.columns(5)

            with col1:
                st.image(os.path.join(imagesPath, predictions[0][0:3], f'{predictions[0]}.jpg'))
            with col2:
                st.image(os.path.join(imagesPath, predictions[1][0:3], f'{predictions[1]}.jpg'))
            with col3:
                st.image(os.path.join(imagesPath, predictions[2][0:3], f'{predictions[2]}.jpg'))        
            with col4:
                st.image(os.path.join(imagesPath, predictions[3][0:3], f'{predictions[3]}.jpg'))
            with col5:
                st.image(os.path.join(imagesPath, predictions[4][0:3], f'{predictions[4]}.jpg')) 
        else:
            st.write('Could not find user with such user id')
            