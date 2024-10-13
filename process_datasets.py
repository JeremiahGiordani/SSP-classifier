import os
import shutil

def process_datasets(root_dir):
    """
    Processes the dataset by flattening class-level folders.
    Moves all '0_real' and '1_fake' images into a single folder under train, test, and val.
    Ensures unique image names by prepending the class folder name to each image.
    """
    # Loop over train, val, and test directories
    for dataset_type in ['train', 'val', 'test']:
        dataset_path = os.path.join(root_dir, dataset_type)

        # Create new target folders for real and fake images
        real_target = os.path.join(dataset_path, '0_real')
        fake_target = os.path.join(dataset_path, '1_fake')

        # Ensure target directories exist
        os.makedirs(real_target, exist_ok=True)
        os.makedirs(fake_target, exist_ok=True)

        # Process each class folder
        for class_folder in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_folder)
            print(f"Processing Class {class_path}")

            # Ensure it's a directory and not the target folders themselves
            if os.path.isdir(class_path) and class_folder not in ['0_real', '1_fake']:
                real_folder = os.path.join(class_path, '0_real')
                fake_folder = os.path.join(class_path, '1_fake')

                # Move images from class/0_real to the target 0_real folder
                if os.path.exists(real_folder):
                    for img in os.listdir(real_folder):
                        # Prepend class folder name to ensure unique filenames
                        new_name = f"{class_folder}_{img}"
                        src = os.path.join(real_folder, img)
                        dst = os.path.join(real_target, new_name)
                        shutil.move(src, dst)

                # Move images from class/1_fake to the target 1_fake folder
                if os.path.exists(fake_folder):
                    for img in os.listdir(fake_folder):
                        # Prepend class folder name to ensure unique filenames
                        new_name = f"{class_folder}_{img}"
                        src = os.path.join(fake_folder, img)
                        dst = os.path.join(fake_target, new_name)
                        shutil.move(src, dst)

                # Optionally, remove the empty class folder
                shutil.rmtree(class_path)

    print("Dataset restructuring complete!")

# Run the function if this file is executed directly
if __name__ == '__main__':
    # Provide the root directory where train, val, and test are located
    root_dir = './dataset'
    process_datasets(root_dir)
