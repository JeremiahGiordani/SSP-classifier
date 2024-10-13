import requests
import os

# Create a directory for the images
os.makedirs("midjourney_images", exist_ok=True)

# API URL
url = "https://datasets-server.huggingface.co/rows"
params = {
    "dataset": "ehristoforu/midjourney-images",
    "config": "default",
    "split": "train",
    "offset": 0,
    "length": 100  # Adjust the length as needed
}

# Fetch the rows
response = requests.get(url, params=params)
data = response.json()

# Download each image
for row in data["rows"]:
    image_url = row["image"]["src"]
    image_name = image_url.split("/")[-1]
    print(f"Downloading {image_name}...")

    # Download the image
    img_data = requests.get(image_url).content
    with open(f"midjourney_images/{image_name}", "wb") as handler:
        handler.write(img_data)

print("All images downloaded!")
