import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from networks.ssp import ssp  # Import your SSP model

# Load the model and weights
def load_model(weights_path):
    model = ssp()  # Initialize the SSP model
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

# Image preprocessing function (same as during training)
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension (1, 3, 256, 256)
    return image

# Function to classify an image
def classify_image(model, image_path):
    # Preprocess the image
    image = preprocess_image(image_path)

    # Run inference
    with torch.no_grad():
        output = model(image)
        probabilities = torch.sigmoid(output).item()  # Get probability

    # Classify based on probability threshold (0.5 for binary classification)
    label = 'Fake' if probabilities > 0.5 else 'Real'
    print(f'Classification: {label}, Probability: {probabilities:.4f}')

# Main function to run the demo
if __name__ == "__main__":
    # Path to the model weights and input image
    weights_path = "./snapshot/sortnet/Net_epoch_best.pth"  # Update if necessary
    image_path = "path/to/your/image.jpg"  # Provide the path to an image

    # Load the model
    model = load_model(weights_path)

    # Classify the input image
    classify_image(model, image_path)
