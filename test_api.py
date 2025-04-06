import requests

url = "http://127.0.0.1:8000/predict"
file_path = "cabbage.jpg"  # Change this to your image path

with open(file_path, "rb") as img_file:
    response = requests.post(url, files={"file": img_file})

print(response.json())  # Output: {"prediction": "Capsicum"}
