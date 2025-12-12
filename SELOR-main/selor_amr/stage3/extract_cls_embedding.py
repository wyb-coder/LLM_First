"""Extract CLS embeddings from BERT for AMR-SELOR (Stage 3 dependency).

【溯源说明】
改造自：extract_base_embedding.py（原 SELOR 嵌入提取脚本）
原文件职能：
  - 使用 load_data() 加载 10% 训练样本
  - 通过 BaseModel 提取 CLS 嵌入
  - 依赖先运行 base.py 训练基础模型
本文件职能：
  - 直接读取指定 CSV 全量样本（绕过 10% 采样）
  - 仅使用预训练 BERT 提取 CLS 嵌入（无需微调）
  - 输出格式与 Stage 3 pretrain_ce_triple.py 兼容
核心改造：
  - 数据源：load_data() 10% 采样 → 直接读取全量 CSV
  - 模型：BaseModel → 直接使用 transformers BERT
  - 依赖：需先运行 base.py → 无依赖，独立运行

Usage:
    python extract_cls_embedding.py \
        --train_csv data/yelp_review_polarity_csv/train_with_amr.csv \
        --text_col 1 \
        --batch_size 32 \
        --out_path result/embeddings/train_cls.pt
"""
import argparse
import os
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--text_col", type=str, default="1", help="Text column name or index")
    parser.add_argument("--csv_has_header", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="HuggingFace model name")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out_path", type=str, default="result/embeddings/train_cls.pt")
    return parser.parse_args()


class TextDataset(Dataset):
    """Simple dataset for text sequences."""
    
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def extract_embeddings(model, dataloader, device):
    """Extract CLS embeddings from BERT."""
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_dim]
            all_embeddings.append(cls_embeddings.cpu())
    
    return torch.cat(all_embeddings, dim=0)


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Load CSV
    header = 0 if args.csv_has_header else None
    df = pd.read_csv(args.train_csv, header=header)
    
    # Get text column
    text_col = int(args.text_col) if str(args.text_col).isdigit() else args.text_col
    texts = df[text_col].tolist()
    print(f"Loaded {len(texts)} samples from {args.train_csv}")
    
    # Load tokenizer and model
    print(f"Loading model: {args.model_name}")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertModel.from_pretrained(args.model_name).to(device)
    
    # Create dataloader
    dataset = TextDataset(texts, tokenizer, args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Extract embeddings
    print("Extracting CLS embeddings...")
    embeddings = extract_embeddings(model, dataloader, device)
    print(f"Embedding shape: {embeddings.shape}")
    
    # Save
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    torch.save(embeddings, args.out_path)
    print(f"Saved to {args.out_path}")


if __name__ == "__main__":
    main()
