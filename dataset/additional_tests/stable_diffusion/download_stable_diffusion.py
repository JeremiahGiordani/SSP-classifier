import pandas as pd
import requests
import os
from PIL import Image
import io

# Step 1: Download the parquet file
def download_parquet(url, output_file):
    print(f"Downloading {url}...")
    response = requests.get(url)
    with open(output_file, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded {output_file} successfully!")

# Step 2: Convert parquet data to images
def convert_parquet_to_images(parquet_file, output_dir):
    print(f"Reading {parquet_file}...")
    # Read parquet data into DataFrame
    df = pd.read_parquet(parquet_file)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if the images are stored as byte strings
    if 'image' not in df.columns:
        raise ValueError("The parquet file does not contain an 'image' column.")

    print(f"Saving images to {output_dir}...")
    for idx, row in df.iterrows():
        try:
            # Convert the byte data to an image
            image_data = row['image']
            img = Image.open(io.BytesIO(image_data))

            # Save the image as PNG (or another format if preferred)
            img_path = os.path.join(output_dir, f'image_{idx}.png')
            img.save(img_path)
            print(f"Saved {img_path}")
        except Exception as e:
            print(f"Failed to process row {idx}: {e}")

    print("All images saved successfully!")

if __name__ == "__main__":
    # Define the download URL and file paths
    PARQUET_URL = "https://huggingface.co/datasets/bitmind/stable-diffusion-xl/resolve/main/data/train-00000-of-00001.parquet"
    PARQUET_FILE = "train-00000-of-00001.parquet"
    OUTPUT_DIR = "stable_diffusion_images"

    # Download the parquet file
    download_parquet(PARQUET_URL, PARQUET_FILE)

    # Convert parquet data to images
    convert_parquet_to_images(PARQUET_FILE, OUTPUT_DIR)
