import requests
import time
import os

URL = "YOUR_CLOUD_RUN_URL/predict" # Replace this!
IMAGE_PATH = "data/test_image.jpg"

print(f"Starting load test on {URL}...")

with open(IMAGE_PATH, 'rb') as f:
    img_data = f.read()

for i in range(50):
    # Send a mix of good and bad requests to test both SLO and Alerts
    files = {'file': ('img.jpg', img_data, 'image/jpeg')}
    try:
        start = time.time()
        r = requests.post(URL, files=files)
        end = time.time()
        print(f"Req {i}: Status {r.status_code} | Latency: {end-start:.2f}s")
    except Exception as e:
        print(f"Error: {e}")
    time.sleep(0.2) # Don't get rate-limited!