from ultralytics import YOLO



model = YOLO("yolov8n.pt")






text = ''

for (k,v) in model.names.items():
	text += f"{k} {v}\n"



with open("labels.txt","w") as f :
	f.write(text)


print("File Created")