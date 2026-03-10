# 🎭 Celebrity Look-Alike Finder

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge\&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge\&logo=streamlit)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?style=for-the-badge\&logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

An **AI-powered application** that identifies which **celebrity you most closely resemble**.
The project uses **Deep Learning** to extract facial features and compare them against a dataset of famous personalities.

---

# 🚀 Features

✨ **Face Detection**
Uses **MTCNN (Multi-task Cascaded Convolutional Networks)** to accurately detect faces in uploaded images.

🧠 **Feature Extraction**
Leverages **VGGFace (ResNet50)** to generate high-dimensional facial embeddings.

🔎 **Similarity Matching**
Uses **Cosine Similarity** to find the closest match between the user's face and the celebrity dataset.

🎨 **Interactive UI**
Built with **Streamlit** for a smooth and responsive web interface.

---

# 🛠️ Tech Stack

| Technology             | Purpose                      |
| ---------------------- | ---------------------------- |
| **Python**             | Core programming language    |
| **Streamlit**          | Interactive web application  |
| **TensorFlow / Keras** | Running deep learning models |
| **VGGFace (ResNet50)** | Facial feature extraction    |
| **MTCNN**              | Face detection               |
| **OpenCV & PIL**       | Image processing             |
| **Scikit-learn**       | Similarity calculation       |
| **Pickle**             | Storage of embeddings        |

---

# 📂 Project Structure

```
WhichCelebrityYouAre/
│
├── app.py                # Main Streamlit application
├── featureExtractor.py   # Generate embeddings for dataset
├── makePickelFile.py     # Create list of image paths
├── test.py               # Testing script
│
├── embeddings.pkl        # Precomputed feature vectors
├── filenames.pkl         # Paths to dataset images
│
├── data/                 # Celebrity image dataset
│   ├── Celebrity1/
│   ├── Celebrity2/
│
└── sample/               # Sample images for testing
```

---

# ⚙️ How It Works

### 1️⃣ Data Pre-processing

* `makePickelFile.py` scans the **data/** folder and stores image paths in `filenames.pkl`.
* `featureExtractor.py` generates embeddings using **VGGFace ResNet50** and saves them in `embeddings.pkl`.

### 2️⃣ User Upload

The user uploads an image through the **Streamlit interface**.

### 3️⃣ Face Detection

**MTCNN** detects the face and crops it from the image.

### 4️⃣ Feature Extraction

The cropped face is resized to **224×224** and passed through the **VGGFace model** to generate an embedding.

### 5️⃣ Similarity Matching

The embedding is compared with stored embeddings using **Cosine Similarity**.

### 6️⃣ Result

The **celebrity with the highest similarity score** is displayed.

---

# 📥 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/abhinavbahadursingh/Which-Celebrity-You-Look-Like.git
cd Which-Celebrity-You-Look-Like
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv .venv
```

Activate environment:

**Windows**

```bash
.venv\Scripts\activate
```

**Mac/Linux**

```bash
source .venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install streamlit tensorflow keras-vggface mtcnn opencv-python scikit-learn pillow tqdm
```

---

# 🖥️ Usage

### Re-generate embeddings (Optional)

If you add new celebrities:

```bash
python makePickelFile.py
python featureExtractor.py
```

### Run the App

```bash
streamlit run app.py
```

---

# 📸 Screenshots

Add screenshots here 👇

```
/screenshots/app_ui.png
/screenshots/result.png
```

Example:

```
![App Screenshot](screenshots/app_ui.png)
```

---

# 💡 Future Improvements

* 🎯 Top 3 celebrity matches
* 📊 Similarity percentage display
* 🌐 Deploy on **Streamlit Cloud**
* 📱 Mobile friendly UI
* 🧠 Larger celebrity dataset

---

# ⚠️ Disclaimer

This project is **for educational purposes only**.
All celebrity images belong to their respective owners.

---

# ⭐ Support

If you like this project:

⭐ **Star the repository**
🍴 **Fork it**
🐛 **Report issues**

---

# 👨‍💻 Author

Developed by **Abhinav Bahadur Singh**

GitHub: https://github.com/abhinavbahadursingh



