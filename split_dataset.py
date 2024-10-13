import os
import random
import shutil

def ensure_dir_exists(dir_path):
    """Ensure that the directory exists."""
    os.makedirs(dir_path, exist_ok=True)

def move_images(src, dst, num_images):
    """Move a specified number of images from the source to the destination."""
    images = os.listdir(src)
    selected_images = random.sample(images, num_images)

    for img in selected_images:
        src_path = os.path.join(src, img)
        dst_path = os.path.join(dst, img)
        shutil.move(src_path, dst_path)

def retain_images(src, retain_count):
    """Retain a specified number of images in the source folder, deleting the rest."""
    images = os.listdir(src)
    if len(images) <= retain_count:
        print(f"Folder {src} already has {retain_count} or fewer images.")
        return

    # Randomly select images to retain
    images_to_retain = random.sample(images, retain_count)

    # Delete all images not in the retained set
    for img in images:
        if img not in images_to_retain:
            os.remove(os.path.join(src, img))

def main():
    # Define the paths
    val_real = 'dataset/val/0_real'
    val_fake = 'dataset/val/1_fake'
    train_real = 'dataset/train/0_real'
    train_fake = 'dataset/train/1_fake'
    test_real = 'dataset/test/0_real'
    test_fake = 'dataset/test/1_fake'

    # Ensure the target directories exist
    ensure_dir_exists(train_real)
    ensure_dir_exists(train_fake)
    ensure_dir_exists(test_real)
    ensure_dir_exists(test_fake)

    # Move 80 real and 80 fake images to the training set
    move_images(val_real, train_real, 80)
    move_images(val_fake, train_fake, 80)

    # Move 10 real and 10 fake images to the test set
    move_images(val_real, test_real, 10)
    move_images(val_fake, test_fake, 10)

    # Retain 10 images in the val folders and delete the rest
    retain_images(val_real, 10)
    retain_images(val_fake, 10)

    print("Dataset splitting and processing complete!")

if __name__ == '__main__':
    main()
