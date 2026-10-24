import io, os, torch, torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from torchvision import transforms, models

BASE = os.path.dirname(os.path.abspath(__file__))

model = models.resnet50()
model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, 2))
model.load_state_dict(torch.load(os.path.join(BASE, "detector.pth"),
                                 map_location="cpu", weights_only=True))
model.eval()

tfm = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

app = FastAPI()

@app.get("/")
def home():
    return FileResponse(os.path.join(BASE, "index.html"))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    with torch.no_grad():
        p = torch.softmax(model(tfm(img).unsqueeze(0)), 1)[0]
    return {"fake": float(p[0]), "real": float(p[1])}   