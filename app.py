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

st.set_page_config(page_title="Celebrity Look-Alike", layout="wide")

st.title("🎭 Celebrity Look-Alike Finder")

# load embeddings
feature_list = np.array(pickle.load(open("embeddings.pkl", "rb")))
filenames = pickle.load(open("filenames.pkl", "rb"))

# load model
model = VGGFace(model="resnet50", include_top=False, input_shape=(224,224,3), pooling="avg")

detector = MTCNN()

uploaded_file = st.file_uploader("Upload your image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)

    # ---- CENTER CHECK BUTTON ----
    c1, c2, c3 = st.columns([1,1,1])
    with c2:
        check = st.button("🔍 Check")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Your Image")
        st.image(img, width=300)

    with col2:
        result_placeholder = st.empty()

    if check:

        results = detector.detect_faces(img_array)

        if len(results) == 0:
            st.error("No face detected")

        else:
            x, y, w, h = results[0]["box"]
            x, y = abs(x), abs(y)

            face = img_array[y:y+h, x:x+w]

            face = Image.fromarray(face)
            face = face.resize((224,224))

            face_array = np.asarray(face).astype("float32")

            expanded = np.expand_dims(face_array, axis=0)
            preprocessed = preprocess_input(expanded)

            result = model.predict(preprocessed).flatten()
            result = result / np.linalg.norm(result)

            similarity = cosine_similarity([result], feature_list)[0]

            index_pos = np.argmax(similarity)

            celebrity_img = cv2.imread(filenames[index_pos])
            celebrity_img = cv2.cvtColor(celebrity_img, cv2.COLOR_BGR2RGB)

            celebrity_name = os.path.basename(os.path.dirname(filenames[index_pos]))

            with result_placeholder.container():

                st.subheader(f"⭐ You look like: **{celebrity_name}**")
                st.image(celebrity_img, width=300)