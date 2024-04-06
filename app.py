import streamlit as st
from PIL import Image,ImageOps
import numpy as np
import tensorflow  as tf
import numpy as np



loaded_model=tf.keras.models.load_model('models/r3.h5')


class_names = ['glioma' , 'meningioma', 'notumor ', 'pituitary']
st.title('Brain Tumor Classification App')


def image_to_array(image_path):
    """
    Converts an image to a numpy array.

    Parameters:
    - image_path: The file path of the image.

    Returns:
    - A numpy array representing the image.
    """
    # Open the image file
    with Image.open(image_path) as img:
        # Convert the image to RGB (in case it's not in RGB format)
        img_rgb = img.convert('RGB')
        # Convert the image into a numpy array
        image_array = np.array(img_rgb)
    
    return image_array

def preprocess_image(image_array):
    """
    Resizes the image to (224, 224, 3) and scales pixel values by 1./255, then converts to uint8.
    
    Parameters:
    - image_array: A numpy array of the image.
    
    Returns:
    - A numpy array of the resized and scaled image with dtype uint8.
    """
    # Ensure the image is a PIL Image for resizing
    image = Image.fromarray(image_array)
    
    # Resize the image
    image_resized = image.resize((224, 224))
    
    # Convert back to numpy array
    image_resized_array = np.asarray(image_resized)
    
    # Scale pixel values and convert to uint8
    image_scaled = (image_resized_array * (1./255)).astype(np.uint8)
    
    return image_scaled


def pred_and_plot(model, filename, class_names):
  """
  Imports an image located at filename, makes a prediction on it with
  a trained model and plots the image with the predicted class as the title.
  """
  # Import the target image and preprocess it
  img = preprocess_image(filename)

  # Make a prediction
  pred = model.predict(tf.expand_dims(img, axis=0))

  # Get the predicted class

  pred_class= class_names[pred.flatten().argmax()] # if more than one output, take the max

def import_and_predict(image_data, model):
        size = (224,224)    
        image = ImageOps.fit(image_data, size)
        img = np.asarray(image)
        img_reshape = img[np.newaxis,...]
        img_f=img_reshape*(1./255)
        prediction = model.predict(img_f)
        return prediction



uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    # Import the target image and preprocess it
    prediction = import_and_predict(image,loaded_model)
    pred_class = class_names[prediction.argmax()]
    # Make a prediction
    #pred = loaded_model.predict(img)
    #y_preds=np.argmax(pred,axis=1)
    # Get the predicted class

    #pred_class = class_names[y_preds.argmax()] # if more than one output, take the max
    
    # Output the prediction
    st.write(f'Prediction: {pred_class}')
