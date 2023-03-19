import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
from tensorflow.keras.utils import img_to_array, load_img

import tensorflow as tf
# Load the TFLite model
model = tf.lite.Interpreter('model_quant.tflite')
model.allocate_tensors()

# Get input and output tensors
input_details = model.get_input_details()
output_details = model.get_output_details()

def predict(image):
    # Load the image
    img = Image.open(image).convert('RGB')
    # Resize the image to (224, 224)
    img = img.resize((224, 224))
    # Convert the image to a numpy array
    img_array = img_to_array(img)
    # Normalize the image
    img_array /= 255.0
    # Add an extra dimension to the image array to match the shape required by the model
    img_array = np.expand_dims(img_array, axis=0)
    # Perform inference on the image using the loaded model
    #pred = model.predict(img_array)
     # Set the input tensor
    model.set_tensor(input_details[0]['index'], img_array)

    # Make prediction using the loaded model
    model.invoke()

    # Get the output tensor and convert to class probabilities
    output_data = model.get_tensor(output_details[0]['index'])
    probabilities = tf.nn.softmax(output_data)[0]

    # Return the predicted class label and confidence score
    class_index = tf.argmax(probabilities)
    class_labels = ['Infected', 'Healthy']
    class_label = class_labels[class_index]
    confidence_score = probabilities[class_index]

    # Return the prediction
    return class_label

def about():
    st.title('About')
    st.write('This is a Streamlit app that predicts whether an image contains a healthy or infected fall army worm. The app uses a machine learning model that was trained on a dataset of fall army worm images.')

def contact():
    st.title('Contact')
    st.write('If you have any questions or comments, please contact us at support@makerere.edu')

def further_research():
    st.title('Further Research')
    st.write('If you are interested in learning more about fall army worms, check out these resources:')

    st.write('- [FAO guide on fall army worm](http://www.fao.org/3/i8019en/i8019en.pdf)')
    st.write('- [CABI guide on fall army worm](https://www.plantwise.org/FullTextPDF/2019/2019pw_ghana_fall_armyworm.pdf)')
    st.write('- [Wikipedia page on fall army worm](https://en.wikipedia.org/wiki/Fall_armyworm)')

def main():
    # Set the title of the app
    st.title('Makerere Fall Army Worm Prediction')

    # Create a list of pages to display
    pages = {
        'Home': predict,
        'About': about,
        'Contact': contact,
        'Further Research': further_research
    }

    # Create a multiselect widget to allow the user to select the page to display
    page = st.sidebar.multiselect('Select a page', list(pages.keys()), default='Home')

    # Call the function corresponding to the selected page
    if page[0] == 'Home':
        image_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
        if image_file is not None:
            result = predict(image_file)
            st.write('Prediction:', result)
    else:
        result = pages[page[0]]()

if __name__ == '__main__':
    main()
