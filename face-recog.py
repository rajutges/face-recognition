import cv2
import numpy as np
import face_recognition
# loading image and convert it in RGB
imgRaju=face_recognition.load_image_file('dataset/raju.jpeg')
imgRaju= cv2.cvtColor(imgRaju , cv2.COLOR_BGR2RGB)
imgTest=face_recognition.load_image_file('dataset/raju test.jpeg')
imgTest= cv2.cvtColor(imgTest , cv2.COLOR_BGR2RGB)
faceLoc=face_recognition.face_locations(imgRaju)[0]
encodeRaju=face_recognition.face_encodings(imgRaju)[0]
cv2.rectangle(imgRaju,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest=face_recognition.face_locations(imgTest)[0]
encodeTest=face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
print("facelOC",faceLoc)

result=face_recognition.compare_faces([encodeRaju],encodeTest)

faceDis=face_recognition.face_distance([encodeRaju],encodeTest)
"""
if(faceDis[0])>0.4:
    result=[False]"""
print(result)
cv2.putText(imgTest,str(result)+str(round(faceDis[0],2)),(50,100),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),2)
cv2.imshow('imgRaju',imgRaju)
cv2.imshow('imgTest',imgTest)
cv2.waitKey(0)
