from ultralytics import YOLO
import cv2
import math ,time
from moviepy.editor import VideoFileClip
import streamlit as st



# model
@st.cache_resource
def load_model():
	st.write( "Model Loading .. "+ str(time.time()))
	return YOLO("yolov8n.pt")


model = load_model()




classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]







@st.cache_data
def write_file(filename,data):
	with open(filename,"w") as f:
		f.write(data)



@st.cache_data
def read_file(filename):
	with open(filename,"r") as f:
		data = f.read()
	return data 




def detect_frame(img):


	results = model(img, stream=True)

	for r in results:

		boxes = r.boxes

		for box in boxes:

			x1, y1, x2, y2 = box.xyxy[0]
			x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

			cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

			confidence = math.ceil((box.conf[0]*100))/100
			#print("Confidence --->",confidence)


			cls = int(box.cls[0])
			#print("Class name -->", classNames[cls])

			cont  = f"Confidence : {confidence}  ClassName : {classNames[cls]}"


			write_file("data.txt",cont)

			org = [x1, y1]
			font = cv2.FONT_HERSHEY_SIMPLEX
			fontScale = 1
			color= (255,0,0)
			thickness = 1

			cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)


	return img








def cut_clip(video_clip,time=5):

	clip = VideoFileClip(video_clip)
	clip = clip.subclip(0,time)

	return clip

	






def record_video(src):


	capture_duration = 15
	path = "output.avi"



	cap = cv2.VideoCapture(src)
	size =  (int(cap.get(3)),int(cap.get(4)))
	fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	out = cv2.VideoWriter(path,fourcc, 10.0, size)


	start_time = time.time()

	while (int(time.time() - start_time) < capture_duration) :
		sucess,frame = cap.read()

		if sucess == True :
			img = detect_frame(frame)

			out.write(img)
			cv2.imshow('Webcam', img)
			if cv2.waitKey(1) == ord('q'):
				break 


	cap.release()
	out.release()
	cv2.destroyAllWindows()



record_video(src="images/cars.avi")