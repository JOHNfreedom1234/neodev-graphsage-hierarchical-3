import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# --- 1. Define Model Architecture (Must match training script) ---
class HierarchicalGraphSAGE(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(HierarchicalGraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_node_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin(x)
        return x

def plot_matrix():
    # --- 2. Load Data and Model ---
    data_dir = "processed"
    if not os.path.exists("best_model.pth"):
        print("Error: 'best_model.pth' not found. Run training first.")
        return

    test_dataset = torch.load(f"{data_dir}/test.pt", weights_only=False)
    meta = torch.load(f"{data_dir}/meta.pt", weights_only=False)
    
    # Setup
    BATCH_SIZE = 16
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    num_node_features = test_dataset[0].x.shape[1]
    class_names = meta["classes"]
    num_classes = len(class_names)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HierarchicalGraphSAGE(num_node_features, 64, num_classes).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=False))
    model.eval()

    # --- 3. Generate Predictions ---
    print("Generating predictions on test set...")
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_trues.extend(data.y_component.cpu().numpy())

    # --- 4. Create Confusion Matrix ---
    # Get unique labels actually present in the test set to avoid empty rows/cols
    unique_labels = sorted(list(set(all_trues) | set(all_preds)))
    target_names = [class_names[i] for i in unique_labels]

    cm = confusion_matrix(all_trues, all_preds, labels=unique_labels)
    
    # Convert to DataFrame for easier plotting with labels
    df_cm = pd.DataFrame(cm, index=target_names, columns=target_names)

    # --- 5. Plot Heatmap ---
    plt.figure(figsize=(10, 8))
    
    # Create the heatmap
    # annot=True writes the numbers in the squares
    # fmt='d' ensures they are integers (not scientific notation)
    # cmap='Blues' gives a clean blue gradient
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    
    plt.title('GraphSAGE Model Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for readability
    plt.yticks(rotation=0)
    
    # Save nicely
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("Success! Confusion matrix saved as 'confusion_matrix.png'")
    # plt.show() # Uncomment if running locally with a display

if __name__ == "__main__":
    plot_matrix()