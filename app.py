import os
import keras
from keras.models import load_model
import streamlit as st 
import tensorflow as tf
import numpy as np

st.set_page_config(page_title="Flower Recognition Model | Salman", page_icon="🌸")
st.header('Flower Classification CNN Model Using TensorFlow v2')

flower_names = [
    'Bougenvillea', 'Bush clock vine', 'Butterfly Pea', 'Cape Jasmine', 'Champak',
    'Dahlia', 'Daisy', 'Dandelion', 'Datura', 'Four o clock flower',
    'Hibiscus', 'Jungle gerinum', 'Mariegold', 'Peacock flower', 'Periwinkle',
    'Potato vine', 'Rose', 'Sunflower', 'Touch me not', 'Tulip'
]

model = load_model('Flower_Recognition_Model_v2.h5')

def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180,180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array,0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = 'The Image belongs to ' + flower_names[np.argmax(result)] + ' with a score of '+ str(np.max(result)*100)
    return outcome

uploaded_file = st.file_uploader('Upload an Image')
if uploaded_file is not None:

    image = uploaded_file.read()     # Directly read the image file into memory
    st.image(uploaded_file, width=200, use_container_width=True)     # Process the image for predictions
    outcome = classify_images(uploaded_file)     # Convert the uploaded file into a format suitable for the model
    st.markdown(outcome)   # Show the result
    
# uploaded_file = st.file_uploader('Upload an Image')
# if uploaded_file is not None:
#     with open(os.path.join('Upload', uploaded_file.name), 'wb') as f:
#         f.write(uploaded_file.getbuffer())
    
#     st.image(uploaded_file, width = 300)

#     st.markdown(classify_images(uploaded_file))



st.write("""
### Flowers this model can detect:
1. Bougainvillea  
2. Bush Clock Vine  
3. Butterfly Pea  
4. Cape Jasmine  
5. Champak  
6. Dahlia  
7. Daisy  
8. Dandelion  
9. Datura  
10. Four O'Clock Flower  
11. Hibiscus  
12. Jungle Geranium  
13. Marigold  
14. Peacock Flower  
15. Periwinkle  
16. Potato Vine  
17. Rose  
18. Sunflower  
19. Touch Me Not  
20. Tulip  
""")
