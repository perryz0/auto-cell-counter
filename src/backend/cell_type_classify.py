import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

class CellTypeDataset(Dataset):
    """
    Custom dataset for cell type descriptions and their corresponding labels.
    Assumes data is preprocessed into text-label pairs.
    """
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class CellTypeClassifier(nn.Module):
    """
    Fine-tune a pre-trained BERT model for cell type classification.
    """
    def __init__(self, num_classes):
        super(CellTypeClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)

    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask)

def train_model(train_data, val_data, num_classes=2, epochs=4, learning_rate=2e-5):
    """
    Train the CellTypeClassifier on the provided training data.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = CellTypeDataset(train_data['texts'], train_data['labels'], tokenizer)
    val_dataset = CellTypeDataset(val_data['texts'], val_data['labels'], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CellTypeClassifier(num_classes)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss}')

        # Validation step
        model.eval()
        total_val_loss = 0
        correct_predictions = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.logits, labels)
                total_val_loss += loss.item()

                _, preds = torch.max(outputs.logits, dim=1)
                correct_predictions += torch.sum(preds == labels)

        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = correct_predictions.double() / len(val_dataset)
        print(f'Validation Loss: {avg_val_loss}, Accuracy: {accuracy}')

    return model

def predict_cell_type(model, text, tokenizer, max_length=128):
    """
    Use the trained model to classify a cell type from a user input string.
    """
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(model.device)
    attention_mask = encoding['attention_mask'].to(model.device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, prediction = torch.max(outputs.logits, dim=1)

    return prediction.item()

# Example usage
if __name__ == "__main__":
    # This section is just for demo purposes; replace with real data
    sample_train_data = {
        "texts": ["cell type 1 description", "cell type 2 description"],
        "labels": [0, 1]  # Label 0 corresponds to type 1, label 1 to type 2, etc.
    }
    sample_val_data = {
        "texts": ["cell type 1 desc", "cell type 2 desc"],
        "labels": [0, 1]
    }

    # Training a new model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = train_model(sample_train_data, sample_val_data, num_classes=2)

    # Classify a new user input
    user_input = "description of cell type 1"
    predicted_label = predict_cell_type(model, user_input, tokenizer)
    print(f"Predicted Cell Type Label: {predicted_label}")
