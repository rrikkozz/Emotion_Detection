import cv2
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def process_frame(frame):
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

  for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

  
    face_region = gray[y:y+h, x:x+w]

   
    analyze = DeepFace.analyze(frame, actions=['emotion', 'age', 'gender'], enforce_detection=False)


    dominant_gender = max(analyze[0]['gender'], key=analyze[0]['gender'].get)  
    gender_probability = analyze[0]['gender'][dominant_gender]


    gender_probability_formatted = "{:.2f}".format(gender_probability)
     


    cv2.putText(frame, f'Emotion: {analyze[0]["dominant_emotion"]}', (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(frame, f'Age: {analyze[0]["age"]}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(frame, f'Gender: {dominant_gender} ({gender_probability_formatted}%)', (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


  cv2.imshow('Face Emotion Detection', frame)


video = cv2.VideoCapture(0)


while True:
  ret, frame = video.read()

  if ret:
    process_frame(frame)


  if cv2.waitKey(1) & 0xFF == ord('q'):
    break


video.release()
cv2.destroyAllWindows()
