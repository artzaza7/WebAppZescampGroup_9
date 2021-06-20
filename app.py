from flask import Flask, render_template, Response, request
import time
import cv2
from openVideo import Video
from Webcam import Video_Web

from scipy.spatial import distance as dist
import numpy as np
import imutils
import cv2
import os

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('Home.html')

@app.route("/Project")
def project():
    return render_template('Project.html')


# ตัวอัปโหลดไฟล์ Video

@app.route("/Video",methods=['GET','POST'])
def video_feed_save():
    file=request.files['fileapp']
    file.save('file.avi')
    return render_template('Video.html')

#Function ดู Video########################################

def gen(openVideo):
    while True:
        frame=openVideo.get_frame()
        yield(b'--frame\r\n'
        b'Content-Type:  image/jpeg\r\n\r\n' + frame +
        b'\r\n\r\n')
        
@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Video()),mimetype='multipart/x-mixed-replace; boundary=frame')

################################################################

@app.route("/Webcam")
def webcam():
    return render_template('Webcam.html')
#Function ดู Webcam ########################################

def genweb(Webcam):
    while True:
        frame=Webcam.get_frame()
        yield(b'--frame\r\n'
        b'Content-Type:  image/jpeg\r\n\r\n' + frame +
        b'\r\n\r\n')
        
@app.route('/webcam_feed')
def webcam_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(genweb(Video_Web()),mimetype='multipart/x-mixed-replace; boundary=frame')

################################################################

if __name__=="__main__":
    app.run(debug=True)