Object Detection App
This project demonstrates object detection using YOLO (You Only Look Once) on both images and videos. It is built using Streamlit for the front end and OpenCV for the computer vision tasks.

Features
Object Detection on Images: Upload an image and the app will detect objects, drawing bounding boxes around them.
Object Detection on Videos: Upload a video and the app will process it, highlighting detected objects in each frame.
Real-Time Object Detection: Perform real-time object detection using your webcam.
Confidence and Threshold Adjustment: Adjust the confidence and threshold settings for more precise object detection.
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/object-detection-app.git
cd object-detection-app
Install the required packages:

Make sure you have Python 3.x installed. Then install the required packages using:

bash
Copy code
pip install -r requirements.txt
Download YOLOv3 Weights and Config files:

You need to download the YOLOv3 weights and config files to run the app. You can do this by running:

python
Copy code
python download_yolo_files.py
Or manually download and place them in the project directory.

Usage
Run the Streamlit app:

bash
Copy code
streamlit run app.py
Select Activity:

Object Detection (Image): Upload an image for object detection.
Object Detection (Video): Upload a video for object detection.
Real-Time Object Detection: Perform object detection in real-time using your webcam.
View and Adjust Results:

After processing, view the detected objects and their confidence levels.
Adjust the confidence and threshold sliders for different detection results.
Model Details
The app uses the following models for object detection:

YOLOv3: Used for detecting objects in images and videos. It is pre-trained on the COCO dataset, which includes 80 different classes.
MobileNetSSD: Used for real-time object detection with webcam feed.
Example Usage
Upload a sample image or video, and the app will detect objects, display bounding boxes, and list the objects detected with confidence levels.
For real-time detection, simply enable your webcam and see the objects detected live.

Requirements
Python 3.x
Streamlit
OpenCV
MoviePy
NumPy
Matplotlib
Pandas
aiortc
Other Python packages (listed in requirements.txt)
License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgments
Streamlit
OpenCV
YOLO
MobileNetSSD
