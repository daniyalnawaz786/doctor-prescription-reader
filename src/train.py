"""Fine-tune TrOCR on the Doctor's Handwritten Prescription dataset."""

import os
import random

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src import config
from src.dataset import PrescriptionDataset, make_collate_fn


def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataframes(base_path):
    train_df = pd.read_csv(os.path.join(base_path, "Training", "training_labels.csv"))
    val_df = pd.read_csv(os.path.join(base_path, "Validation", "validation_labels.csv"))
    return train_df, val_df


def train(base_path):
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = TrOCRProcessor.from_pretrained(config.BASE_MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(config.BASE_MODEL_NAME).to(device)

    # Required config for training a VisionEncoderDecoder from scratch heads
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    train_df, val_df = load_dataframes(base_path)

    train_ds = PrescriptionDataset(
        train_df, os.path.join(base_path, "Training", "training_words"), processor
    )

    collate_fn = make_collate_fn(processor)
    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    model.train()

    for epoch in range(config.NUM_EPOCHS):
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in progress:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        print(f"Epoch {epoch} average loss: {total_loss / len(train_loader):.4f}")

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.MODEL_DIR)
    processor.save_pretrained(config.MODEL_DIR)
    print(f"Model saved to {config.MODEL_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path",
        required=True,
        help="Path to the extracted 'Doctor's Handwritten Prescription BD dataset' folder",
    )
    args = parser.parse_args()
    train(args.base_path)
