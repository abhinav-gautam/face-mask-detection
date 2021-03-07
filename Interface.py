import cv2
import numpy as np
from keras.models import load_model
model=load_model("model2-009.model")
results={0:'without mask',1:'mask'}
color_dict={0:(0,0,255),1:(0,255,0)}
rect_size = 4
cap = cv2.VideoCapture(0) 
fourcc = cv2.VideoWriter_fourcc(*'DIVX') 
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640 , 480)) 
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
    _, image = cap.read()
    image=cv2.flip(image,1,1) 
    
    rerect_size = cv2.resize(image, (image.shape[1] // rect_size, image.shape[0] // rect_size))
    faces = face_classifier.detectMultiScale(rerect_size)
    for face in faces:
        x, y, w, h = [v * rect_size for v in face] 
        
        face_img = image[y:y+h, x:x+w]
        face_img_resized=cv2.resize(face_img,(150,150))
        face_img_normalized=face_img_resized/255.0
        face_img_reshaped=np.reshape(face_img_normalized,(1,150,150,3))
        result=model.predict(face_img_reshaped)
        
        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(image,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(image,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(image, results[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    out.write(image)
    cv2.imshow('LIVE',image)
    key = cv2.waitKey(10)
    
    if key == 13: # Enter Key 
        break
cap.release()
out.release()
cv2.destroyAllWindows()