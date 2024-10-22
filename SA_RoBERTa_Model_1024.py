import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_scheduler,
)
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm.auto import tqdm

df = pd.read_csv("dataset.csv")

# Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["sentence"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)
print(f"Training samples: {len(train_texts)}, Validation samples: {len(val_texts)}")


# Dataset Class
class SentimentDataset(Dataset):
    def __init__(self, sentences, labels):
        tokenizer = RobertaTokenizer.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment",
            clean_up_tokenization_spaces=True,
        )
        self.encodings = tokenizer(
            sentences, truncation=True, padding=True, max_length=150
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# DataLoaders for Train and Validation
train_dataset = SentimentDataset(train_texts, train_labels)
val_dataset = SentimentDataset(val_texts, val_labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Fine-Tuning
model = RobertaForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment", num_labels=3
)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# Training Loop
model.train()
progress_bar = tqdm(range(num_training_steps))
train_losses = []
val_losses = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        running_loss += loss.item()
        progress_bar.update(1)
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}")

    # Validation phase
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"].to(device)
            outputs = model(**batch)
            loss = outputs.loss
            val_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted"
    )

    print(
        f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}"
    )
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    model.train()

# Training and validation loss curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.show()

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Neutral", "Positive"],
    yticklabels=["Negative", "Neutral", "Positive"],
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
