import io
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from torchvision import transforms
from model import get_model

app = FastAPI(title="MLOps PyTorch Model Serving API")

# Initialize global variables
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the image transformations (must match the val_dataset transforms in dataset.py)
image_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    )
])

@app.on_event("startup")
def load_checkpoint():
    """Loads the model checkpoint into memory when the server starts."""
    global model
    try:
        # Re-instantiate the model architecture
        model = get_model(architecture="resnet18", num_classes=10)
        
        # Load the weights from the mounted checkpoint directory
        checkpoint_path = "/app/checkpoints/classifier_v1.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load the state dictionary
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        print("Model checkpoint loaded successfully.")
    except Exception as e:
        print(f"Failed to load model checkpoint: {e}")

@app.get("/health")
def health_check():
    """Returns 200 if the model is successfully loaded into memory."""
    if model is not None:
        return {"status": "healthy"}
    raise HTTPException(status_code=503, detail="Model is not loaded.")

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """Accepts an image upload and returns class probabilities."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        # Read the uploaded image bytes
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Transform the image and add a batch dimension
        input_tensor = image_transform(img).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(input_tensor)
            # Apply softmax to convert raw logits to probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
        # Convert tensor to a standard Python list for the JSON response
        probs_list = probabilities[0].tolist()
        
        return {"class_probabilities": probs_list}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")