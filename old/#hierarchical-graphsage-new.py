import os
import random
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from sklearn.metrics import classification_report, f1_score, confusion_matrix # Added confusion_matrix
import numpy as np
import pandas as pd # We use pandas just for the pretty matrix print

# --- 1. The Model Architecture ---
class HierarchicalGraphSAGE(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(HierarchicalGraphSAGE, self).__init__()
        # Layer 1: Aggregates neighbor info (Attribute -> Element)
        self.conv1 = SAGEConv(num_node_features, hidden_channels)
        # Layer 2: Aggregates deeper context (Element -> Component structure)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        # Layer 3: Final refinement
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        # Classifier head
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Node Embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = x.relu()
        
        x = self.conv3(x, edge_index)

        # 2. Global Pooling (Readout)
        x = global_mean_pool(x, batch)

        # 3. Classifier
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin(x)
        return x

# --- 2. Training Setup ---

def train():
    data_dir = "processed"
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' not found. Run preprocessing first.")
        return

    train_dataset = torch.load(f"{data_dir}/train.pt", weights_only=False)
    val_dataset = torch.load(f"{data_dir}/val.pt", weights_only=False)
    test_dataset = torch.load(f"{data_dir}/test.pt", weights_only=False)
    meta = torch.load(f"{data_dir}/meta.pt", weights_only=False)

    BATCH_SIZE = 16
    HIDDEN_CHANNELS = 64
    LEARNING_RATE = 0.001
    EPOCHS = 100
    
    num_node_features = train_dataset[0].x.shape[1]
    num_classes = len(meta["classes"])
    class_names = meta["classes"]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HierarchicalGraphSAGE(num_node_features, HIDDEN_CHANNELS, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)

    # Weighted Loss Setup
    all_train_labels = [data.y_component.item() for data in train_dataset]
    class_counts = np.bincount(all_train_labels, minlength=num_classes)
    weights = len(all_train_labels) / (num_classes * (class_counts + 1e-6))
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    print(f"Starting training on {device}...")
    best_val_f1 = 0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y_component)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch)
                pred = out.argmax(dim=1)
                preds.extend(pred.cpu().numpy())
                trues.extend(data.y_component.cpu().numpy())
        
        val_f1 = f1_score(trues, preds, average='macro', zero_division=0)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {total_loss/len(train_loader):.4f} | Val F1 (Macro): {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "best_model.pth")

    # --- FINAL EVALUATION ---
    print("\n" + "="*60)
    print(" EVALUATION RESULTS ")
    print("="*60)
    
    model.load_state_dict(torch.load("best_model.pth", weights_only=False))
    model.eval()
    
    all_preds, all_trues = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_trues.extend(data.y_component.cpu().numpy())

    # Get unique labels present in THIS test run
    unique_labels = sorted(list(set(all_trues) | set(all_preds)))
    target_names = [class_names[i] for i in unique_labels]

    # 1. Classification Report
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(all_trues, all_preds, labels=unique_labels, target_names=target_names, zero_division=0))

    # 2. Confusion Matrix
    print("\n--- CONFUSION MATRIX ---")
    print("(Rows = Actual, Columns = Predicted)\n")
    
    cm = confusion_matrix(all_trues, all_preds, labels=unique_labels)
    
    # Use Pandas for a readable table
    df_cm = pd.DataFrame(cm, index=target_names, columns=target_names)
    
    # Print with tabulate-style formatting purely via pandas to string
    print(df_cm.to_string())
    
    print("\n" + "="*60)
    
    # Optional: Save it for your thesis
    df_cm.to_csv("confusion_matrix.csv")
    print("Confusion matrix saved to 'confusion_matrix.csv'")

if __name__ == "__main__":
    train()