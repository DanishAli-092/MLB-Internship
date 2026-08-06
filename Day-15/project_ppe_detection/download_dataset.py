from roboflow import Roboflow
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("ROBOFLOW_API_KEY")

rf = Roboflow(api_key=api_key)
project = rf.workspace("sdp-lfigk").project("ppe-detection-ozhfb")
dataset = project.version(14).download("yolov11")

print("dataset download ")