import streamlit as st
import numpy as np
import pickle
import cv2
import os
from mtcnn import MTCNN
from PIL import Image
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# 1. Page Configuration
st.set_page_config(page_title="Celebrity Look-Alike", layout="wide")

# 2. CSS to hide "Deploy" button, Header, and Footer
hide_style = """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stAppDeployButton {display: none;}
    </style>
"""
st.markdown(hide_style, unsafe_allow_html=True)

# 3. Load Data & Model (Caching use kiya hai taaki baar-baar load na ho)
@st.cache_resource
def load_models():
    model = VGGFace(model="resnet50", include_top=False, input_shape=(224,224,3), pooling="avg")
    detector = MTCNN()
    return model, detector

@st.cache_data
def load_features():
    feature_list = np.array(pickle.load(open("embeddings.pkl", "rb")))
    filenames = pickle.load(open("filenames.pkl", "rb"))
    return feature_list, filenames

model, detector = load_models()
feature_list, filenames = load_features()

# 4. UI Elements
st.title("🎭 Celebrity Look-Alike Finder")
uploaded_file = st.file_uploader("Upload your image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Image loading
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)

    # UI Layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Your Image")
        st.image(img, use_container_width=True)
        
        # Center Check Button
        if st.button("🔍 Check"):
            with col2:
                # Face Detection
                results = detector.detect_faces(img_array)

                if len(results) == 0:
                    st.error("No face detected! Please use a clearer photo.")
                else:
                    # Cropping & Preprocessing
                    x, y, w, h = results[0]["box"]
                    x, y = abs(x), abs(y)
                    face = img_array[y:y+h, x:x+w]
                    
                    # Resize to VGG Input size
                    face_img = Image.fromarray(face).resize((224, 224))
                    face_array = np.asarray(face_img).astype("float32")
                    
                    # Expand Dims & Preprocess
                    expanded = np.expand_dims(face_array, axis=0)
                    preprocessed = preprocess_input(expanded)

                    # Feature Extraction
                    result = model.predict(preprocessed).flatten()
                    
                    # Cosine Similarity Calculation
                    # [Image of cosine similarity formula]
                    similarity = cosine_similarity([result], feature_list)[0]
                    index_pos = np.argmax(similarity)

                    # Get Celebrity Image and Name
                    celeb_path = filenames[index_pos]
                    celebrity_img = cv2.imread(celeb_path)
                    celebrity_img = cv2.cvtColor(celebrity_img, cv2.COLOR_BGR2RGB)
                    
                    # Extract Name from folder (Assuming folder name is the celebrity name)
                    celebrity_name = os.path.basename(os.path.dirname(celeb_path)).replace('_', ' ')

                    # Display Result
                    st.subheader(f"⭐ You look like: **{celebrity_name}**")
                    st.image(celebrity_img, use_container_width=True)
                    st.success(f"Similarity Score: {round(similarity[index_pos]*100, 2)}%")

else:
    st.info("Please upload an image to start.")
