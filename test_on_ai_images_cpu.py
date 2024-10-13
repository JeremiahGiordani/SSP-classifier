import os
import torch
from utils.util import set_random_seed
from utils.tdataloader import get_single_loader
from networks.ssp import ssp
from tqdm import tqdm
from options import TrainOptions

def load_model(model_path):
    """Load the trained model from a given path."""
    model = ssp().cuda()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print(f'Model loaded from {model_path}')
    return model

def evaluate(loader, model):
    """Evaluate the model on a specific loader."""
    correct_images = 0
    total_images = len(loader.dataset)
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Testing', unit='batch'):
            preds = torch.sigmoid(model(images)).ravel()
            correct_images += (((preds > 0.5) & (labels == 1)) |
                               ((preds < 0.5) & (labels == 0))).sum().item()
    
    accuracy = correct_images / total_images
    return accuracy

def run_test(test_path, subdir, image_dir, model_path):
    """Main function to run the test."""
    print(f'Testing on dataset: {test_path}')
    
    # Configure test options
    opt = TrainOptions().parse(print_options=False)
    opt.image_root = test_path  # Set the test dataset path

    # Load the data
    ai_loader, _ = get_single_loader(opt, subdir, image_dir, False) # Assuming single dataset for testing
    
    # Load the model
    model = load_model(model_path)

    # Evaluate on AI and Nature images
    print('Evaluating AI-generated images...')
    ai_accuracy = evaluate(ai_loader, model)
    print(f'AI Accuracy: {ai_accuracy:.4f}')

if __name__ == '__main__':
    # Set random seed for reproducibility
    set_random_seed()

    # Hard-code the paths for testing
    TEST_DATASET_PATH = './datasets'  # Change this path if needed
    MODEL_PATH = 'snapshot/ssp/Net_epoch_best.pth'
    SUBDIR = 'additional_tests/midjourney_jpg'
    IMAGEDIR = 'midjourney_images'

    # Ensure the model file exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f'Model not found at {MODEL_PATH}')

    # Run the test
    run_test(TEST_DATASET_PATH, SUBDIR, IMAGEDIR, MODEL_PATH)
