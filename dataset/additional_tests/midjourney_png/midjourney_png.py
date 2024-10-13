import requests
import os

# Hugging Face dataset API endpoint
DATASET_URL = "https://datasets-server.huggingface.co/rows"
DATASET_NAME = "saq1b/midjourney-v6.1"
CONFIG = "default"
SPLIT = "train"
OFFSET = 0
LENGTH = 100  # Adjust as needed to download more files

# Function to get dataset rows
def get_dataset_rows(offset=0, length=100):
    url = f"{DATASET_URL}?dataset={DATASET_NAME}&config={CONFIG}&split={SPLIT}&offset={offset}&length={length}"
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch dataset rows: {response.status_code}")

    return response.json()

# Function to download PNG images only
def download_png_images(rows):
    os.makedirs("downloaded_images", exist_ok=True)  # Create directory for images

    for row in rows["rows"]:
        for key, value in row["row"].items():
            if isinstance(value, str) and value.endswith(".png"):
                filename = os.path.join("downloaded_images", os.path.basename(value))
                download_image(value, filename)

# Helper function to download and save an image
def download_image(url, filename):
    print(f"Downloading {url} to {filename}...")
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(filename, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
    else:
        print(f"Failed to download {url}: {response.status_code}")

if __name__ == "__main__":
    print("Fetching dataset rows...")
    dataset_rows = get_dataset_rows(OFFSET, LENGTH)
    print("Starting PNG download...")
    download_png_images(dataset_rows)
    print("Download complete.")
