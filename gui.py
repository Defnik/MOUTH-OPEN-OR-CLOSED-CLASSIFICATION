import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image,ImageTk
import tensorflow as tf 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json

#load your trained model
def FacialExpressionModel(json_file,weights_file):
    with open(json_file,"r") as file:
        loaded_model_json=file.read()
        model=model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    return model

model=FacialExpressionModel("model_a.json","model_mouth.h5")

#predicting config
def predict_mouth(filepath):
    img=image.load_img(filepath,target_size=(48,48))
    img_array=image.img_to_array(img)
    img_array=np.expand_dims(img_array,axis=0)
    prediction=model.predict(img_array)

    if prediction[0][0] >0.5:
        return "mouth open"
    else:
        return "mouth closed"
    
#opening folder for uploading image
def open_image():
    global image
    file_path=filedialog.askopenfilename()
    if file_path:
        result=predict_mouth(file_path)
        label_result.config(text=result)

        #display the selected image
        img=Image.open(file_path)
        img=img.resize((48,48))
        img=ImageTk.PhotoImage(img)
        label_image.config(image=img)
        label_image.image=img
        label_image['image']=img


#creating tkinter window
root=tk.Tk()
root.title("MOUTH PREDICTION")
root.geometry("800x600")
root.configure(background='#CDCDCD')
#creating a button to an image
button_open=tk.Button(root,text="open Image",command=open_image,padx=10,pady=5)
button_open.configure(background='#364156',foreground='white',font=('arial',15,'bold'))
button_open.pack()
#creating a label for the image
label_image=tk.Label(root,image="")
label_image.pack(side='bottom',expand='True')
#creating a label to display result
label_result=tk.Label(root,text="")
label_result.configure(background='#364156',foreground='white',font=('arial',20,'bold'))
label_result.pack(side='bottom',pady=30)
#To run the tkinter loop
root.mainloop()
