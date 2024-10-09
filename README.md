Emotion Detection with DeepFace and OpenCV

This project uses DeepFace and OpenCV to detect faces in a live video feed and analyze the dominant emotion, estimated age, and gender of the detected faces in real time. The project highlights the power of deep learning in computer vision by providing real-time predictions on facial expressions and characteristics.

Features

Detect faces in a live video stream using OpenCV.
Analyze facial emotions, age, and gender using DeepFace.
Display the detected face with a bounding box and show the detected emotion, age, and gender with confidence.
Technologies Used

OpenCV: For face detection and handling video frames.
DeepFace: For facial attribute analysis, including emotion, age, and gender.
Installation

Prerequisites
Ensure you have Python installed. You will need the following libraries:

OpenCV: Install OpenCV using pip:
bash
Copy code
pip install opencv-python
DeepFace: Install DeepFace using pip:
bash
Copy code
pip install deepface
Additional Dependencies: You might also need to install tensorflow and keras for DeepFace to work properly:
bash
Copy code
pip install tensorflow keras
Setup
Clone the repository and navigate to the project directory:

bash
Copy code
git clone https://github.com/yourusername/emotion-detection.git
cd emotion-detection
Usage

Ensure you have a working webcam.
Run the Python script:
bash
Copy code
python emotion_detection.py
The program will access your webcam and start detecting faces. For each detected face, it will display:
Emotion: The dominant emotion on the face (e.g., happy, sad, neutral).
Age: Estimated age of the person.
Gender: Predicted gender with probability.
Press q to quit the video stream.
Code Explanation

Face Detection: The project uses the Haar Cascade Classifier from OpenCV to detect faces in real-time.
python
Copy code
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
Emotion, Age, and Gender Analysis: Once a face is detected, DeepFace is used to analyze it and return attributes like emotion, age, and gender.
python
Copy code
analyze = DeepFace.analyze(frame, actions=['emotion', 'age', 'gender'], enforce_detection=False)
Displaying Results: The emotion, age, and gender are displayed on the video feed above the detected face.
python
Copy code
cv2.putText(frame, f'Emotion: {analyze[0]["dominant_emotion"]}', (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
Sample Output

The bounding box is drawn around the detected face.
Text annotations appear showing the detected emotion, age, and gender above the face.
Future Improvements

Add support for detecting multiple faces simultaneously.
Enhance the model's performance with custom training datasets.
Implement additional facial attributes, such as race or personality traits.
License

This project is licensed under the MIT License - see the LICENSE file for details.
