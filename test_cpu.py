import os
import torch
from utils.util import set_random_seed
from utils.tdataloader import get_test_loader
from networks.ssp import ssp
from tqdm import tqdm
from options import TrainOptions

def load_model(model_path):
    """Load the trained model from a given path."""
    model = ssp()
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

def run_test(test_path, model_path):
    """Main function to run the test."""
    print(f'Testing on dataset: {test_path}')
    
    # Configure test options
    opt = TrainOptions().parse(print_options=False)
    opt.image_root = test_path  # Set the test dataset path

    # Load the data
    test_loader = get_test_loader(opt) # Assuming single dataset for testing
    
    # Load the model
    model = load_model(model_path)

    # Evaluate on AI and Nature images
    print('Evaluating AI-generated images...')
    ai_accuracy = evaluate(test_loader['test_ai_loader'], model)
    print(f'AI Accuracy: {ai_accuracy:.4f}')

    print('Evaluating Nature (Real) images...')
    nature_accuracy = evaluate(test_loader['test_nature_loader'], model)
    print(f'Nature Accuracy: {nature_accuracy:.4f}')

    # Calculate and display total accuracy
    total_accuracy = (ai_accuracy * test_loader['ai_size'] + 
                      nature_accuracy * test_loader['nature_size']) / \
                     (test_loader['ai_size'] + test_loader['nature_size'])

    print(f'Total Accuracy: {total_accuracy:.4f}')

if __name__ == '__main__':
    # Set random seed for reproducibility
    set_random_seed()

    # Hard-code the paths for testing
    TEST_DATASET_PATH = 'dataset'  # Change this path if needed
    MODEL_PATH = 'snapshot/ssp/Net_epoch_best.pth'

    # Ensure the model file exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f'Model not found at {MODEL_PATH}')

    # Run the test
    run_test(TEST_DATASET_PATH, MODEL_PATH)
