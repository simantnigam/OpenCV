import tkinter as tk
from tkinter.ttk import Button
from tk import *
from PIL import Image, ImageTk
import os
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import threading


emote_model = Sequential()
emote_model.add(Conv2D(32, kernel_size=(3, 3),
                activation='relu', input_shape=(48, 48, 1)))
emote_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emote_model.add(MaxPooling2D(2, 2))  # downsampling
emote_model.add(Dropout(0.25))  # prevent overfitting,train more on noise
emote_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emote_model.add(MaxPooling2D(2, 2))
emote_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emote_model.add(MaxPooling2D(2, 2))
emote_model.add(Dropout(0.25))
emote_model.add(Flatten())
emote_model.add(Dense(1024, activation='relu'))  # hidden layer
emote_model.add(Dropout(0.5))
emote_model.add(Dense(7, activation='softmax'))

emote_model.load_weights('model.h5')
cv2.ocl.setUseOpenCL(False)


emotion_dict = {0: 'Angry', 1: 'Disgusted', 2: 'Fearful',
                3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprised'}

cur_path = os.path.dirname(os.path.abspath(__file__))

emoji_dict = {0: cur_path+'/data/emoji/angry.png', 1: cur_path+'/data/emoji/disgust.png', 2: cur_path+'/data/emoji/fear.png', 3: cur_path +
              '/data/emoji/happy.png', 4: cur_path+'/data/emoji/neutral.png', 5: cur_path+'/data/emoji/sad.png', 6: cur_path+'/data/emoji/surprise.png'}


global last_frame1
global cap
cap = cv2.VideoCapture(0)
global frame_number
last_frame1 = np.zeros((480, 640, 1), dtype=np.uint8)
show_text = [0]


def show_subject():
    #cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Can't open the camera")
    global frame_number

    flag1, frame1 = cap.read()
    frame1 = cv2.resize(frame1, (600, 500))
    bounding_box = cv2.CascadeClassifier(
        'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(
        frame1, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y+h, x:x+w]
        cropped_img = np.expand_dims(np.expand_dims(
            cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        prediction = emote_model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame1, emotion_dict[maxindex], (x+20, y-60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        show_text[0] = maxindex
    if flag1 is None:
        print('Image not read properly!!!')
    elif flag1:
        global last_frame1
        last_frame1 = frame1.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        root.update()
        lmain.after(10, show_subject)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
        # exit()


def show_avatar():
    frame2 = cv2.imread(emoji_dict[show_text[0]])
    pic2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    img2 = Image.fromarray(pic2)
    imgtk2 = ImageTk.PhotoImage(image=img2)
    lmain2.imgtk2 = imgtk2
    lmain3.configure(
        text=emotion_dict[show_text[0]], font=('arial', 45, 'bold'))
    lmain2.configure(image=imgtk2)
    root.update()
    lmain2.after(10, show_avatar)


if __name__ == "__main__":
    frame_number = 0
    root = tk.Tk()
    lmain = tk.Label(master=root, padx=50, bd=10)
    lmain2 = tk.Label(master=root, bd=10)
    lmain3 = tk.Label(master=root, bd=10, fg='#CDCDCD', bg='black')
    lmain.pack(side='left')
    lmain.place(x=50, y=250)
    lmain3.pack()
    lmain3.place(x=1000, y=250)
    lmain2.pack(side='right')
    lmain2.place(x=900, y=350)

    root.title('Face Emotion to Emoji')
    root.geometry('1400x900+100+10')
    root['bg'] = 'black'
    exitButton = tk.Button(root, text='Quit', fg='red',
                           command=root.destroy, font=('arial', 25, 'bold'))
    exitButton.pack(side='bottom')

    threading.Thread(target=show_subject).start()
    threading.Thread(target=show_avatar).start()
    root.mainloop()
    cap.release()
