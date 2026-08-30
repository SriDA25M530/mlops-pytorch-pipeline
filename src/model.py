import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_model(architecture: str, num_classes: int) -> nn.Module:
    """
    Returns a PyTorch model based on the specified architecture.
    """
    if architecture == "resnet18":
        # Load a pre-trained ResNet-18 model
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Modify the final fully connected layer for the target number of classes
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model
    else:
        raise ValueError(f"Architecture '{architecture}' is not supported.")