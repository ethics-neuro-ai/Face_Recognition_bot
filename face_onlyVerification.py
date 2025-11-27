#!/usr/bin/env python
# coding: utf-8

# In[118]:



import numpy as np
from deepface.commons import distance as dst
import cv2
from facenet_pytorch import MTCNN
from facenet_pytorch import InceptionResnetV1
import torch
import torchvision.transforms as transforms
import base64
import tempfile
import re
from bson import Binary
from skimage.transform import resize
from io import BytesIO
from PIL import Image
from torchvision.transforms import ToTensor
from pymongo import MongoClient
import base64
import face_recognition
import sys
from sklearn.preprocessing import LabelEncoder
from  telegram import Bot
import redis
import json
from deepface import DeepFace


client = MongoClient('mongodb+srv://faziolistella:B0qlsJucolNe1aJT@cluster0.lk0oteg.mongodb.net/?retryWrites=true&w=majority')

database = client["test"]
collection = database["users"] 

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, (160, 160))  # Resize to 160x160
    image = (image - 127.5) / 128.0 # Normalize pixel values to [-1, 1] .. prova image = (image - 127.5) / 128.0
    return image


# 

# In[6]:


def extract_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    #image = cv2.imread(image_path)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_img = image[y:y+h, x:x+w]
        return face_img
    else:
        return None


# In[111]:


model_dir = '/Users/stellafazioli/Downloads/app/app/models/modelUpdated_DSet.xml'
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(model_dir)





 

# In[119]:


if __name__ == '__main__':
    # Check if the captured image data is provided as a command-line argument
    if len(sys.argv) > 1:
        captured_image_data = sys.argv[1]
        name=sys.argv[2]
        
        
    else:
        print("No captured image data provided.")
        


# In[120]:



# In[ ]:
#def load_image_from_base64(base64_data):
#    decoded_data = base64.b64decode(base64_data)
#    image = Image.open(BytesIO(decoded_data))
#    return image
#
#image = load_image_from_base64(captured_image_data)

#image_data = captured_image_data['data']
image_data = base64.b64decode(captured_image_data)
print(captured_image_data)
try:
    
    
    # Convert bytes to image
    image = Image.open(BytesIO(image_data))
    # Resize image
    target_shape = (160, 160)
    resized_image = image.resize(target_shape)
    # Convert image to array
    image_array = np.array(resized_image)
    # Save the image to a file
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg')
    cv2.imwrite(temp_file.name, image_array)
    if image_array is not None and image_array.size > 0:
        try:
            face=extract_face(image_array)
            processed_image=preprocess_image(face)
            print(processed_image.shape)
            #face_encoding=face_recognition.face_encodings(face)
            #print(face_encoding)
            #encoding_array.append(face_encoding)
        except:
            print("Error: Failed to crop the face from the image.")
        else:
            print("Error: Failed to load the image.")
except base64.binascii.Error as e:
    print("Error: Invalid base64-encoded string:", str(e))






#def recognize_face(image):
#    if len(face.shape) > 2 and face.shape[2] > 1:
#        gray_image = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#    else:
#        gray_image = face
#    
#    label, confidence = face_recognizer.predict(gray_image)
#    return label, confidence
#

def get_all_encodings():
    # Load all encodings from MongoDB
    encodings = []
    names = []
    for doc in collection.find():
        encoding_list = doc['encoding']
        name = doc['name']
        for encoding_dict in encoding_list:
            embedding = encoding_dict
            if isinstance(embedding, list):
                encoding = np.array(embedding, dtype=np.float32)
                encodings.append(encoding)
                names.append(name)
    return encodings, names

def compute_distance_matrix(image):
    encodings, names = get_all_encodings()

    # Extract face embeddings using Facenet model
    target_shape = (160, 160)
    resized_image = cv2.resize(image, target_shape)
    embedding = DeepFace.represent(resized_image, model_name='Facenet', enforce_detection=False)

    if embedding is not None:
        similarity_matrix = []
        for encoding in encodings:
            similarity_score_ = dst.findCosineDistance(embedding, encoding)
            similarity_score = similarity_score_['cosine']
            similarity_matrix.append(similarity_score)
        similarity_matrix = np.array(similarity_matrix)
        return similarity_matrix, names
    else:
        return None, None

def perform_verification(image, user_id, threshold):
    similarity_matrix, names = compute_distance_matrix(image)
    
    if similarity_matrix is not None:
        user_index = names.index(user_id)
        user_scores = similarity_matrix[user_index]
        user_name = names[user_index]
        
        verification_result = user_scores < threshold  # Compare scores to threshold return true or false
        
        return verification_result, user_name
    else:
        return None, None




redis_client = redis.Redis(host='localhost', port=6379)
threshold = dst.findThreshold('FaceNet512', 'cosine')

verification_,name_=perform_verification(image,name,threshold)
if verification_:
    message = f"Face recognized as {name} "
    print(message)   
    sys.stdout.flush()
    redis_client.publish('message_channel', json.dumps(message))
else:
     message = f"Face not recognized as {name} "
     print(message)   
     sys.stdout.flush()
     redis_client.publish('message_channel', json.dumps(message))



