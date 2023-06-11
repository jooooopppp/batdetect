import torch
from utils.analysis import BatCallCNN

class CustomModel(torch.nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        # Define your model structure here. It needs to be the same as the one used in batdetect2

    def forward(self, x):
        # Define the forward pass here
        pass


def load_model(model_path):
    # Create an instance of your model
    model = BatCallCNN()  # Replace 'YourModelClass' with the actual class name of your model

    model_state_dict = model.state_dict()
    torch.save({'model_state_dict': model_state_dict}, model_path)

    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    # Load the model state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])

    # Ensure the model is in evaluation mode
    model.eval()

    return model
