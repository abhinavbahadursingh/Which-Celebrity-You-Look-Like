# Celebrity look-alike finder

from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from mtcnn import MTCNN
from PIL import Image

# load embeddings and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# load model
model = VGGFace(model='resnet50', include_top=False, input_shape=(224,224,3), pooling='avg')

# face detector
detector = MTCNN()

# load image
sample_img = cv2.imread('sample/OIP.jpg')

# detect face
results = detector.detect_faces(sample_img)

if len(results) == 0:
    print("No face detected")
    exit()

x, y, width, height = results[0]['box']
x, y = abs(x), abs(y)

face = sample_img[y:y+height, x:x+width]

# preprocess face
image = Image.fromarray(face)
image = image.resize((224,224))

face_array = np.asarray(image)
face_array = face_array.astype('float32')

expanded_face_array = np.expand_dims(face_array, axis=0)
preprocessed_img = preprocess_input(expanded_face_array)

# extract features
result = model.predict(preprocessed_img).flatten()

# normalize feature
result = result / np.linalg.norm(result)

# compare with stored embeddings
similarity = cosine_similarity([result], feature_list)[0]

# find best match
index_pos = np.argmax(similarity)

# show matched celebrity image
temp_img = cv2.imread(filenames[index_pos])

cv2.imshow("Input Image", sample_img)
cv2.imshow("Most Similar Celebrity", temp_img)
cv2.waitKey(0)
cv2.destroyAllWindows()