import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from keras.models import load_model
import cv2
from streamlit_drawable_canvas import st_canvas

# Load the pre-trained generator model
model = load_model('generator_modelkaggle.h5')

# Define function to preprocess the image
def preprocess_image(image):
    image = image.resize((256, 256))  # Resize to match model input size
    image = image.convert("RGB")  # Convert grayscale image to RGB (3 channels)
    image = np.array(image).astype('float32')
    image = (image / 127.5) - 1  # Normalize to [-1, 1]
    image = np.expand_dims(image, axis=0)   # Add batch dimension
    return image

# Function to post-process the output
def postprocess_output(output):
    output = (output[0] + 1) * 127.5  # De-normalize to [0, 255]
    output = output.astype(np.uint8)
    return output

# Streamlit UI
st.title("Pix2Pix Edge-to-Shoes Demo")

# Instructions
st.write("Draw a sketch of a shoe on the canvas, and the model will generate an image of the shoe.")

# Set up columns for the canvas and output
col1, col2 = st.columns(2)

with col1:
    st.write("### Sketch your shoe")
    # Create a canvas where the user can draw
    canvas_result = st_canvas(
        stroke_width=2,
        stroke_color="black",
        background_color="white",
        width=256,
        height=256,
        drawing_mode="freedraw",
        key="canvas"
    )

with col2:
    st.write("### Model Output")
    if 'output_image' in st.session_state:
        st.image(st.session_state['output_image'], width=256, caption="Generated Shoe")
    else:
        st.write("Model output will appear here")

# Button to send the sketch to the model
if st.button("Generate Shoe"):
    if canvas_result.image_data is not None:
        sketch_image = Image.fromarray(canvas_result.image_data.astype('uint8'))
        preprocessed_image = preprocess_image(sketch_image)

        # Get model prediction
        generated_image = model.predict(preprocessed_image)

        # Post-process the output
        output_image = postprocess_output(generated_image)

        # Save output image in session state to display it on the UI
        st.session_state['output_image'] = output_image

        # Show the generated image
        with col2:
            st.image(output_image, width=256, caption="Generated Shoe")
    else:
        st.warning("Please draw a sketch first.")
