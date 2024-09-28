import deeplake
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from flask import Flask, request, jsonify
from PIL import Image
import io

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} backend")

# Data transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    )
])

# Load DeepLake dataset and split
ds = deeplake.load('hub://activeloop/wiki-art', read_only=True)
class_names = ds.labels.info.class_names
num_classes = len(class_names)

ds.transforms = {'images': data_transforms}
train_ds, val_ds = ds.random_split([0.8, 0.2])

# Create DataLoaders
train_loader = train_ds.pytorch(
    tensors=['images', 'labels'],
    batch_size=64,
    shuffle=True,
    num_workers=0  # Set to 0 or adjust based on your system
)
val_loader = val_ds.pytorch(
    tensors=['images', 'labels'],
    batch_size=64,
    shuffle=False,
    num_workers=0
)

# Define the LightningModule
class WikiArtModel(LightningModule):
    def __init__(self, num_classes):
        super(WikiArtModel, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(
            self.model.fc.in_features, num_classes
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=0.001
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch['images'], batch['labels'].squeeze().long()
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = torch.sum(preds == labels).float() / len(labels)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch['images'], batch['labels'].squeeze().long()
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = torch.sum(preds == labels).float() / len(labels)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return {'preds': preds.cpu(), 'labels': labels.cpu()}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([o['preds'] for o in outputs])
        labels = torch.cat([o['labels'] for o in outputs])
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='weighted'
        )
        metrics = {
            'val_accuracy': accuracy,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1
        }
        self.log_dict(metrics, prog_bar=True)

# Initialize model
model = WikiArtModel(num_classes=num_classes)

# Training with PyTorch Lightning
trainer = pl.Trainer(
    max_epochs=5,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1
)
trainer.fit(model, train_loader, val_loader)

# Save the trained model
trainer.save_checkpoint("wikiart_model.ckpt")

# Flask app for serving predictions
app = Flask(__name__)

# Inference transforms
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    )
])

# Load the model for inference
inference_model = WikiArtModel(num_classes=num_classes)
inference_model.load_state_dict(torch.load("wikiart_model.ckpt")['state_dict'])
inference_model.eval()
inference_model.to(device)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = inference_transforms(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = inference_model(image)
            _, predicted = torch.max(outputs, 1)
        predicted_style = class_names[predicted.item()]
        return jsonify({"predicted_style": predicted_style})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True)
