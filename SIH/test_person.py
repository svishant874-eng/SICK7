# pyrefly: ignore [missing-import]
from ultralytics import YOLO

# Model load
model = YOLO(r"C:\Users\MANJEET\Desktop\SICK7\models\person_best.pt")

# Input video
input_video = r"C:\Users\MANJEET\Desktop\SICK7\SIH\stock-footage-busy-pedestrian-street-crowd-people-walking-on-around-city-square-view-from-the-top-aerial-shoot.webm"

# Detection
model.predict(
    source=input_video,
    conf=0.25,
    save=True,
    project=r"C:\Users\MANJEET\Desktop\SICK7\SIH\runs",
    name="person_video_test"
)

print("Detection complete!")