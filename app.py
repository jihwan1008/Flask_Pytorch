import io
import json
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch
from flask import Flask, jsonify, render_template, request
import os

app = Flask(__name__)

net = models.resnet34()
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 2)
device = torch.device('cpu')
checkpoint = torch.load('best.pth', map_location=device)
net.load_state_dict(checkpoint['net'])
net.eval()

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize((224,224)),
                                        #transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.5, 0.5, 0.5],
                                            [0.5, 0.5, 0.5])])
    image = Image.open(io.BytesIO(image_bytes))

    return my_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):

    idx = str()
    prob = float()
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    softmax = F.softmax(outputs).detach().cpu()
    _, predicted = outputs.max(1)
    
    if predicted.numpy()[0] == 1:
        idx = 'Gingivitis'
        prob = softmax.numpy()[0][1]
    else:
        idx = 'Normal'
        prob = softmax.numpy()[0][1]

    return idx, prob

@app.route('/favicon.ico') 
def favicon():     
    return send_from_directory(os.path.join(app.root_path, 'static'),                                'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/', methods=['GET'])
def root:
    return jsonify({'msg' : 'Try POSTing to the /predict endpoint with RGB image attachment'})

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        diagnosis, probability = get_prediction(image_bytes=img_bytes)

        return jsonify({'Diagnosis': diagnosis, 'Probability': probability})

if __name__ == '__main__':
    app.run()
