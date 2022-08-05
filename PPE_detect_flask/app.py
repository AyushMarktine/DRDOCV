
from flask import Flask, redirect, render_template, request, session, Response, url_for
import pandas as pd
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#from pandas.io.json import json_normalize
#import csv
import os

import torch
from werkzeug.utils import secure_filename
from PIL import Image


#model = torch.hub.load('ultralytics/yolov5', 'custom', path = "best.pt")

UPLOAD_FOLDER = os.path.join('staticfiles', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__, template_folder='templatesFiles', static_folder='staticfiles')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'You Will Never Guess'

def detect_object(uploaded_image_path):
    #img = cv2.imread(uploaded_image_path)[..., ::-1] 
    # img = cv2.resize(img, (640, 640))
    model = torch.hub.load('ultralytics/yolov5', 'custom', path = "best.pt")
    img = model(Image.open(uploaded_image_path))
    #img= model(img, size=640)
    #img.show()
    #output_image_path = os.path.join('staticfiles','res', 'output_image.jpg')
    os.chdir('staticfiles')
    #cv2.imwrite('output_image.jpg', img)
    #print(app.config['STATIC_FOLDER'])

    #img.save(os.path.join(app.root_path,"staticfiles/res/output_image.jpg"))
    img.save("rest/output_image.jpg", "JPEG")
    #img.save('rest')
    s = os.path.split(uploaded_image_path)
    img_name= s[1]
    output_image_path = os.path.join('staticfiles','JPEG', img_name)

    return(output_image_path)

 
@app.route('/')
def index():
    return render_template('index_upload_and_display_image.html')
 
@app.route('/',  methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        uploaded_img = request.files['uploaded-file']
        print("path ",uploaded_img)
        img_filename = secure_filename(uploaded_img.filename)
        print("img ",img_filename)
        uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
 
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
 
        return render_template('index_upload_and_display_image_page2.html')
 
@app.route('/show_image')
def displayImage():
    img_file_path = session.get('uploaded_img_file_path', None)
    return render_template('show_image.html', user_image = img_file_path)
 
@app.route('/detect_object')
def detectObject():
    uploaded_image_path = session.get('uploaded_img_file_path', None)
    output_image_path = detect_object(uploaded_image_path)
    print(output_image_path)
    #im_path= session.get('output_image_path', None)
    
    return render_template('show_image2.html', user_image = output_image_path)
    
    #return redirect(url_for('esult', filename='PPE.jpg'))



@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

if __name__=='__main__':
    app.run(debug = True)
    #print(dir(app))
