import os
import shutil

def move_png_files(source_dir, target_dir):
    """Move all PNG files from the source directory to the target directory."""
    os.makedirs(target_dir, exist_ok=True)  # Create target directory if it doesn't exist

    # Walk through the source directory and its subdirectories
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".png"):
                # Construct full file path
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_dir, file)

                print(f"Moving {source_file} to {target_file}...")
                shutil.move(source_file, target_file)

    print("All PNG files have been moved successfully.")

if __name__ == "__main__":
    # Set the source and target directories
    source_directory = os.path.join(os.getcwd(), "midjourney-v6.1")  # Current directory where the script is running (midjourney-v6.1)
    target_directory = os.path.join(os.getcwd(), "all_midjourney_images")

    # Move PNG files
    move_png_files(source_directory, target_directory)
