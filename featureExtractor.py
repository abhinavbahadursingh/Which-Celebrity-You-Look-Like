# for this we will use VGG face ( basically usko break krke ussko feature extract krne ke liye use krenge )

from tensorflow.keras.preprocessing import image
from keras_vggface import VGGFace
from keras_vggface.utils import preprocess_input
import numpy as np
import pickle
from tqdm import tqdm

filenames = pickle.load(open('filenames.pkl' , 'rb'))

# abhi model building krenge VGGFace model ( internally resnet50 cnn model use krenge ) top layer hata denge iska
model = VGGFace(model='resnet50', include_top=False, input_shape=(224,224,3) , pooling='avg')

def feature_extractor(img_path , model):
    img = image.load_img(img_path, target_size=(224, 224)) #VGGFace model 224×224 input expect karta hai.
    img_arr = image.img_to_array(img) # Model sirf numbers (arrays) samajhta hai.
    expanded_img = np.expand_dims(img_arr, axis=0) # Model batch input expect karta hai.
    preprocessed_img = preprocess_input(expanded_img) # Model training ke time jo format tha same format me input dena padta hai.

    model_output = model.predict(preprocessed_img).flatten()

    return model_output


features = []
for filename in tqdm(filenames):
    features.append(feature_extractor(filename , model))

pickle.dump(features, open('embeddings.pkl', 'wb'))