import os
import timm
from transformers import BertModel
import torch.nn as nn
from tqdm import tqdm
from process_data import MyData
from torch.utils.data import DataLoader
import torch
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
MASTER_CSV = "adidas_dataset/labels.csv"
if not os.path.isfile(MASTER_CSV):
    raise FileNotFoundError(MASTER_CSV)
df = pd.read_csv(MASTER_CSV, sep=";")
label_map = {label: idx for idx, label in enumerate(df["label"].unique())}
train_dataset = MyData(MASTER_CSV, "adidas_dataset", label_map, split="train")
test_dataset = MyData(MASTER_CSV, "adidas_dataset", label_map, split="test")


train_data = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=32,
    pin_memory=True,
    drop_last=True,
)
test_data = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=32,
    pin_memory=True,
    drop_last=True,
)

print("hi")


def args():
    parser = argparse.ArgumentParser(description="Training script for multimodal model")
    parser.add_argument(
        "--model-train",
        type=str,
        default="Single_Text",
        help="Model to use: Q_cons_fusion or MLP_fusion",
    )
    parser.add_argument(
        "--epochs", type=int, default=15, help="Number of training epochs"
    )
    return parser.parse_args()


class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")

    def forward(self, input_ids, attention_mask):
        # tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.bert.device)
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state[:, 0, :]


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.image_encoder = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.image_encoder.head = nn.Identity()

    def forward(self, img):
        return self.image_encoder(img)


class Single_Text(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, len(label_map))
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        txt = self.text_encoder(input_ids, attention_mask)
        x = self.relu(self.fc1(txt))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)


class Single_Image(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, len(label_map))
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, img):
        feats = self.image_encoder(img)
        x = self.relu(self.fc1(feats))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)


def test(model, dataloader, args=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            image = batch["image"].to(device)
            if args.model_train == "Single_Text":
                output = model(input_ids, attention_mask)
            else:
                output = model(image)
            predictions = output.argmax(dim=1)  # Get predicted labels
            correct += (predictions == label).sum().item()
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")
    return correct / len(test_dataset), precision, recall, f1


def train(model, dataloader, epochs=10, args=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=0.01)

    scaler = GradScaler("cuda")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        all_labels = []
        all_preds = []
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            # img = batch['image'].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            img = batch["image"].to(device)
            optimizer.zero_grad()
            with autocast("cuda"):
                if args.model_train == "Single_Text":
                    output = model(input_ids, attention_mask)
                else:
                    output = model(img)
                # print(output.size())

                loss = criterion(output, label)
            # print(loss)
            prediction = output.argmax(dim=1)
            correct += (prediction == label).sum().item()
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(prediction.cpu().numpy())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        precision = precision_score(all_labels, all_preds, average="macro")
        recall = recall_score(all_labels, all_preds, average="macro")
        f1 = f1_score(all_labels, all_preds, average="macro")
        result_test, pr1, re1, f11 = test(model, test_data, args)

        log_message = f"Epoch {epoch + 1} / {epochs}: \n Train: Loss: {total_loss / len(dataloader):.4f} Accuracy: {correct / len(train_dataset):.4f},  Recall: {recall}, Precision: {precision}, F1-Score: {f1} \n Test :  Test_accuracy: {result_test:.4f},  Precision: {pr1}, Recall: {re1}, F1-Score: {f11} \n\n"
        print(log_message)

        # Save to log file
        if args.model_train == "Single_Text":
            with open("single_txt.txt", "a") as f:
                f.write(log_message + "\n")
        else:
            with open("single_img.txt", "a") as f:
                f.write(log_message + "\n")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = args()
    if args.model_train == "Single_Text":
        model = Single_Text()
        model = torch.nn.DataParallel(model)
        model.to(device)
        train(model, train_data, epochs=args.epochs, args=args)
    else:
        model = Single_Image()
        print(args.model_train)
        model = torch.nn.DataParallel(model)
        model.to(device)
        train(model, train_data, epochs=args.epochs, args=args)
