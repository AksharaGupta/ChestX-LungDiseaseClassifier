# Import flask which will handle our communication with the frontend
# Also import few other libraries
from flask import Flask, render_template, request
from scipy.misc import imread, imresize, imsave
from keras.preprocessing.image import  load_img, img_to_array
import numpy as np
import re
import sys
import base64
import os
# Path to our saved model
sys.path.append(os.path.abspath("./model"))
from load import *



# Initialize flask app
app = Flask(__name__)
#Initialize some global variables
global model, graph
model, graph = init()
def convertImage(imgData1):
  imgstr = re.search(r'base64,(.*)', str(imgData1)).group(1)
  with open('output.png', 'wb') as output:
    output.write(base64.b64decode(imgstr))
@app.route('/')
def login():
  return render_template("login.html")
@app.route('/home')
def index():
  return render_template("index.html")
@app.route('/loginfalse')
def loginfalse():
  return render_template("loginfalse.html")
@app.route('/form')
def form():
  return render_template("form.html")
@app.route('/predict/', methods=['GET', 'POST'])
def predict():
  # Predict method is called when we push the 'Predict' button 
  # on the webpage. We will feed the user drawn image to the model
  # perform inference, and return the classification
  imgData = request.get_data()
  convertImage(imgData)
  # # read the image into memory
  # # # make it the right size
  # x = imresize(x, (450, 150))
  # x = x.reshape(1, 150, 150, 3)
  x = load_img('output.png', target_size=(150,150))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  diseases = ['Cardiomegaly','Emphysema','Effusion','Hernia','Nodule','Pneumothorax','Atelectasis','Pleural_Thickening','Mass','Edema','Consolidation','Infiltration','Fibrosis','Pneumonia','No Finding']
  diseases.sort()
  with graph.as_default():
    out = model.predict(x)
    result=out[0]
    response = np.argmax(result)
    return diseases[response]
  
if __name__ == "__main__":
# run the app locally on the given port
  app.run(host='0.0.0.0', port=5000)