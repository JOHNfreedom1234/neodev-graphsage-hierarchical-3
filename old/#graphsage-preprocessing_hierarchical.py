import json
import os
from collections import defaultdict, Counter

import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# --- MAPPING CONSTANTS ---
EDGE_TYPE_MAP = {
    0: "Parent -> Child",
    1: "Child -> Parent",
    2: "Sibling <-> Sibling"
}

# --- Helper Functions ---

def get_root_node(item):
    """Safely extracts root from 'dom' or 'contents' and handles list wrappers."""
    root = item.get("dom") or item.get("content") or item.get("contents")
    if isinstance(root, list) and len(root) > 0:
        return root[0]
    return root

def collect_tags_from_dataset(dataset):
    tags = set()
    def dfs(node):
        if not node or not isinstance(node, dict): return
        tag = (node.get("tag") or node.get("type") or node.get("name") or "").lower()
        if tag: tags.add(tag)
        for child in node.get("children", []): dfs(child)
    for item in dataset:
        root = get_root_node(item)
        if root: dfs(root)
    return sorted(tags)

def is_interactive(tag, attrs):
    return "href" in attrs or "onclick" in attrs or tag in {"button", "input", "select", "textarea", "a"}

def is_media(tag, attrs):
    return "src" in attrs or tag in {"img", "video", "audio", "svg"}

def traverse_dom(root):
    nodes, edges = [], []
    def dfs(node, parent_idx, depth):
        if not node or not isinstance(node, dict): return
        idx = len(nodes)
        tag = (node.get("tag") or node.get("type") or node.get("name") or "").lower()
        attrs = node.get("attributes", {})
        children = node.get("children", [])
        
        nodes.append({
            "tag": tag, "attrs": attrs, "depth": depth, "parent": parent_idx,
            "children": [], "is_leaf": len(children) == 0,
            "is_interactive": is_interactive(tag, attrs), "is_media": is_media(tag, attrs),
        })
        
        if parent_idx is not None:
            # RELATIONSHIPS
            edges.append((parent_idx, idx, 0)) # Type 0: Parent -> Child
            edges.append((idx, parent_idx, 1)) # Type 1: Child -> Parent
            nodes[parent_idx]["children"].append(idx)
            
        for c in children: dfs(c, idx, depth + 1)
        
    dfs(root, None, 0)
    
    # Sibling Relationships (Type 2)
    for n in nodes:
        kids = n["children"]
        for i in range(len(kids) - 1):
            a, b = kids[i], kids[i + 1]
            edges.append((a, b, 2))
            edges.append((b, a, 2))
    return nodes, edges

def compute_subtree_stats(nodes):
    if not nodes: return
    def dfs(idx):
        desc, inter, media = 0, 0, 0
        for c in nodes[idx]["children"]:
            d, i, m = dfs(c)
            desc += 1 + d
            inter += i + int(nodes[c]["is_interactive"])
            media += m + int(nodes[c]["is_media"])
        nodes[idx].update({"descendants": desc, "interactive_desc": inter, "media_desc": media})
        return desc, inter, media
    dfs(0)

def attribute_flags(attrs):
    return [int("class" in attrs), int("id" in attrs), int("style" in attrs),
            int("href" in attrs), int("src" in attrs), int("role" in attrs),
            int(any(k.startswith("aria-") for k in attrs))]

def build_features(nodes, tag2id, tfidf, scaler):
    css_texts, numeric_feats, binary_feats, tag_ids = [], [], [], []
    max_depth = max(n["depth"] for n in nodes) + 1e-6
    
    for n in nodes:
        tag_ids.append(tag2id.get(n["tag"], tag2id["__unk__"]))
        css_texts.append(n["attrs"].get("class", ""))
        numeric_feats.append([n["depth"], n["depth"]/max_depth, len(n["children"]),
                             n["descendants"], n["interactive_desc"], n["media_desc"]])
        binary_feats.append([int(n["is_leaf"]), int(n["is_interactive"]), int(n["is_media"]), *attribute_flags(n["attrs"])])
        
    css_vec = tfidf.transform(css_texts).toarray()
    num_scaled = scaler.transform(np.array(numeric_feats))
    x = np.concatenate([css_vec, num_scaled, np.array(binary_feats)], axis=1)
    return torch.tensor(x, dtype=torch.float), torch.tensor(tag_ids, dtype=torch.long)

def build_graph(item, tag2id, tfidf, scaler, comp_enc):
    root = get_root_node(item)
    if not root: return None
    nodes, edges = traverse_dom(root)
    compute_subtree_stats(nodes)
    x, tag_ids = build_features(nodes, tag2id, tfidf, scaler)
    if len(edges) > 0:
        edge_index = torch.tensor([[u, v] for u, v, _ in edges], dtype=torch.long).t()
        edge_type = torch.tensor([t for _, _, t in edges], dtype=torch.long)
    else:
        # Create an empty 2D tensor of shape [2, 0]
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)
    y_component = torch.tensor(comp_enc.transform([item["label"]]), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_type=edge_type, tag_ids=tag_ids, y_component=y_component)

def split_dataset(data, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Correctly performs a three-way split."""
    train_graphs, temp_graphs = train_test_split(data, train_size=train_ratio, random_state=seed, shuffle=True)
    # The remaining ratio is (1 - train_ratio). We want val_ratio of the TOTAL.
    # So we take (val_ratio / remaining_ratio) of the temporary set.
    remaining_ratio = 1.0 - train_ratio
    val_relative_ratio = val_ratio / remaining_ratio
    val_graphs, test_graphs = train_test_split(temp_graphs, train_size=val_relative_ratio, random_state=seed, shuffle=True)
    return train_graphs, val_graphs, test_graphs

# --- Main Preprocessing ---

def preprocess(json_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # 1. Vocab & Global Stats
    tags = ["__unk__"] + collect_tags_from_dataset(raw)
    tag2id = {t: i for i, t in enumerate(tags)}
    css_corpus, all_numeric_feats, labels_list = [], [], []
    all_edge_types = Counter()

    print(f"Analyzing {len(raw)} components and their elements...")
    for item in raw:
        root = get_root_node(item)
        if not root: continue
        labels_list.append(item["label"])
        nodes, edges = traverse_dom(root)
        for _, _, etype in edges: all_edge_types[etype] += 1
        compute_subtree_stats(nodes)
        max_depth = max(n["depth"] for n in nodes) + 1e-6
        for n in nodes:
            css_corpus.append(n["attrs"].get("class", ""))
            all_numeric_feats.append([n["depth"], n["depth"]/max_depth, len(n["children"]),
                                     n["descendants"], n["interactive_desc"], n["media_desc"]])

    # 2. Fit Encoders
    tfidf = TfidfVectorizer(max_features=128).fit(css_corpus)
    scaler = StandardScaler().fit(np.array(all_numeric_feats))
    comp_enc = LabelEncoder().fit(labels_list)

    # 3. Build Graphs
    graphs = [g for item in raw if (g := build_graph(item, tag2id, tfidf, scaler, comp_enc))]

    # 4. Corrected Split Call
    train_g, val_g, test_g = split_dataset(graphs)

    # 5. Save
    torch.save(train_g, os.path.join(out_dir, "train.pt"))
    torch.save(val_g, os.path.join(out_dir, "val.pt"))
    torch.save(test_g, os.path.join(out_dir, "test.pt"))
    torch.save({
        "tag2id": tag2id, 
        "classes": list(comp_enc.classes_),
        "edge_type_map": EDGE_TYPE_MAP
    }, os.path.join(out_dir, "meta.pt"))

    # --- HIERARCHICAL REPORT ---
    print("\n" + "="*60)
    print(" HIERARCHICAL PREPROCESSING REPORT")
    print("="*60)
    print(f"Total Components (Graphs): {len(graphs)}")
    print(f"Total Elements (Nodes):     {len(all_numeric_feats)}")
    print("-" * 60)
    print("RELATIONSHIP DISTRIBUTION (Attribute/Element Hierarchy):")
    for etype, name in EDGE_TYPE_MAP.items():
        print(f"  Type {etype} ({name:20}): {all_edge_types[etype]} links")
    print("-" * 60)
    print(f"{'COMPONENT LABEL':<25} | {'COUNT':<5} | {'ID'}")
    print("-" * 60)
    for idx, name in enumerate(comp_enc.classes_):
        print(f"{name:<25} | {labels_list.count(name):<5} | {idx}")
    print("="*60)

if __name__ == "__main__":
    preprocess(json_path="labeled_data2.json", out_dir="processed")