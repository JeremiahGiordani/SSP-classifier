import os
import io
import requests
from PIL import Image

# Define the output directory
output_dir = "stable_diffusion_images"
os.makedirs(output_dir, exist_ok=True)

# Hugging Face dataset API endpoint and parameters
DATASET_URL = "https://datasets-server.huggingface.co/rows"
DATASET_NAME = "bitmind/stable-diffusion-xl"
CONFIG = "default"
SPLIT = "train"
OFFSET = 0
LENGTH = 100  # Adjust this as needed

def get_dataset_rows(offset=0, length=100):
    """Fetch dataset rows from the Hugging Face API."""
    url = f"{DATASET_URL}?dataset={DATASET_NAME}&config={CONFIG}&split={SPLIT}&offset={offset}&length={length}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch dataset rows: {response.status_code}")
    return response.json()

def save_images_from_rows(rows):
    """Process and save images from dataset rows."""
    for idx, row in enumerate(rows["rows"]):
        try:
            # Inspect the row structure to see what 'image' contains
            print(f"Row {idx} content: {row}")

            # Check if 'image' key exists and contains valid data
            if 'image' in row['row']:
                image_url = row['row']['image']['src']  # Adjust this if needed
                img_data = requests.get(image_url).content

                # Open and save the image
                img = Image.open(io.BytesIO(img_data))
                img_path = os.path.join(output_dir, f'image_{idx}.png')
                img.save(img_path)
                print(f"Saved {img_path}")
            else:
                print(f"No image data found in row {idx}")

        except Exception as e:
            print(f"Failed to process row {idx}: {e}")

if __name__ == "__main__":
    print("Fetching dataset rows...")
    dataset_rows = get_dataset_rows(OFFSET, LENGTH)
    print("Processing and saving images...")
    save_images_from_rows(dataset_rows)
    print("All images processed.")
