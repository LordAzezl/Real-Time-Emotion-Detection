'''
Emotion Detection Using AI
'''
import json
import spotipy
import webbrowser
import csv
import random
import tkinter
from keras.models import load_model
from time import sleep
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import spotipy
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
classifier = load_model('./model_optimal.h5')
label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    print(ret, frame)
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # make a prediction on the ROI, then lookup the class

            preds = classifier.predict(roi)[0]
            # print("\nprediction = ", preds)
            label = class_labels[preds.argmax()]
            # print("\nprediction max = ", preds.argmax())
            # print("\nlabel = ", label)
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        # print("\n\n")
    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


if (label == "Happy"):
    file, file2 = (open("./SongsList\\EngHappy.csv", "r")),  (open("./SongsList\\HindiHappy.csv", "r"))
    lst = list(csv.reader(file, delimiter=","))
    lst2 = list(csv.reader(file2, delimiter=","))
    for i in lst2:
        lst.append(i)
    song = lst[random.randint(1,100)]


elif(label =="Sad"):
    file, file2 = (open("./SongsList\\EngSad.csv", "r")), (open("./SongsList\\HindiSad.csv", "r"))
    lst = list(csv.reader(file, delimiter=","))
    lst2 = list(csv.reader(file2, delimiter=","))
    for i in lst2:
        lst.append(i)
    song = lst[random.randint(1, 100)]


else:
    file = open("./SongsList\\Allsongs.csv", "r")
    lst = list(csv.reader(file, delimiter=","))
    song = lst[random.randint(1, 100)]


username = '31dvyjojayyrpqdrtijqqbdes6zm'
clientID = 'a13449deaf12435cb5e031a90883ebe7'
clientSecret = 'bedb5741770547958cc9714205d3b760'
redirect_uri = 'http://localhost:8888/callback'
oauth_object = spotipy.SpotifyOAuth(clientID, clientSecret, redirect_uri)
token_dict = oauth_object.get_access_token()
token = token_dict['access_token']
spotifyObject = spotipy.Spotify(auth=token)
user_name = spotifyObject.current_user()

# To print the JSON response from
# browser in a readable format.
# optional can be removed
print(json.dumps(user_name, sort_keys=True, indent=4))

while True:
    print("Welcome to the project, " + user_name['display_name'])
    print("0 - Exit the console")
    print("1 - Next song")
    user_input = int(input("Enter Your Choice: "))
    if user_input == 1:
        results = spotifyObject.search(song[0], 1, 0, "track")
        songs_dict = results['tracks']
        song_items = songs_dict['items']
        song = song_items[0]['external_urls']['spotify']
        webbrowser.open(song)
        print('Song has opened in your browser.')
    elif user_input == 0:
        print("Good Bye, Have a great day!")
        break
    else:
        print("Please enter valid user-input.")