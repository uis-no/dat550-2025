from PIL import Image
import requests
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch
from torch.optim import AdamW

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog", "cats and remotes"], images=image, return_tensors="pt", padding="max_length", max_length=77, truncation=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
print(probs)


import requests
import torch
from transformers import CLIPProcessor, CLIPModel, CLIPConfig

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(requests.get(item["url"], stream=True).raw)
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        input_ids = self.processor(text=item["text"], return_tensors="pt", padding="max_length", max_length=77, truncation=True).input_ids
        label = torch.tensor(item["label"])
        return {
            "pixel_values": pixel_values.squeeze(),
            "input_ids": input_ids.squeeze(),
            "label": label
        }

train_data = [
    {"url": "http://images.cocodataset.org/val2017/000000039769.jpg", "text": "a photo of cats", "label": 1},
    {"url": "http://images.cocodataset.org/val2017/000000039769.jpg", "text": "a photo of animals", "label": 1},
    {"url": "http://images.cocodataset.org/val2017/000000039769.jpg", "text": "a photo of two humans", "label": 0},
    {"url": "http://images.cocodataset.org/val2017/000000039769.jpg", "text": "a photo of cars", "label": 0},
    {"url": "http://images.cocodataset.org/train2017/000000106140.jpg", "text": "a photo of a car", "label": 0},
]

val_data = [
    {"url": "http://images.cocodataset.org/train2017/000000106140.jpg", "text": "a photo of a cat", "label": 0},
    {"url": "http://images.cocodataset.org/train2017/000000106140.jpg", "text": "a photo of an airplane", "label": 1},
    # Add more test data points as needed
]

test_data = [
    {"url": "http://images.cocodataset.org/train2017/000000106140.jpg", "text": "a photo of a cat", "label": 0},
    {"url": "http://images.cocodataset.org/train2017/000000106140.jpg", "text": "a photo of an airplane", "label": 1},
    # Add more test data points as needed
]

train_dataset = CLIPDataset(train_data)
val_dataset = CLIPDataset(val_data)
test_dataset = CLIPDataset(test_data)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

# Load the pre-trained CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Define the fine-tuning configuration
config = CLIPConfig(
    text_config=model.text_model.config.to_dict(),
    vision_config=model.vision_model.config.to_dict(),
    projection_dim=512,
    logit_scale_init_value=2.6592,
)

# Instantiate the fine-tuned CLIP model
fine_tuned_model = CLIPModel(config)

# Freeze the pre-trained model parameters
for param in fine_tuned_model.parameters():
    param.requires_grad = False

# Define the fine-tuning head
# fine_tuned_model.classification_head = torch.nn.Linear(config.projection_dim, 2)
fine_tuned_model.classification_head = torch.nn.Linear(1024, 1)
# Define the optimizer and training loop
optimizer = AdamW(fine_tuned_model.classification_head.parameters(), lr=5e-5)
loss_fn = torch.nn.BCEWithLogitsLoss()

for epoch in range(10):
    fine_tuned_model.train()
    train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = fine_tuned_model(pixel_values=batch["pixel_values"], input_ids=batch["input_ids"])
        combined_embeds = torch.cat((outputs.image_embeds, outputs.text_embeds), dim=1)
        logits = fine_tuned_model.classification_head(combined_embeds)
        
        loss = loss_fn(logits.view(-1), batch["label"].float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Print average training loss per epoch
    print(f"Epoch {epoch+1}, Loss: {train_loss / len(train_loader)}")
    
    # Evaluation phase
    fine_tuned_model.eval()
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for batch in val_loader:
            outputs = fine_tuned_model(pixel_values=batch["pixel_values"], input_ids=batch["input_ids"])
            combined_embeds = torch.cat((outputs.image_embeds, outputs.text_embeds), dim=1)
            logits = fine_tuned_model.classification_head(combined_embeds)
        
            preds = torch.sigmoid(logits.view(-1)) > 0.5  # Get binary predictions
            total_correct += (preds == batch["label"]).sum().item()
            total_count += preds.size(0)
    
    # Print validation accuracy
    print(f"Validation Accuracy: {total_correct / total_count}")
    
train_dataset = load_dataset("gokuls/coco_dataset", split='train')
val_dataset = load_dataset("gokuls/coco_dataset", split='train')
test_dataset = load_dataset("gokuls/coco_dataset", split='train')

for data in val_dataset:
    image = Image.open(requests.get(data["url"], stream=True).raw)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    input_ids = processor(text=data["text"], return_tensors="pt", padding="max_length", max_length=77, truncation=True).input_ids
    label = torch.tensor(data["label"])
    print(data, label)
    break
