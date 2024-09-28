from ultralytics import YOLO

model = YOLO('../Model/best.pt')

result = model.predict(source = '../C2.jpg', imgsz=640,conf=0.3, save=True, show = True )