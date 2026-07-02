"""PyTorch Dataset and collate function for the prescription images."""

import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class PrescriptionDataset(Dataset):
    """Maps a labels dataframe to (pixel_values, labels) pairs for TrOCR."""

    def __init__(self, df, image_dir, processor, text_col="MEDICINE_NAME"):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.processor = processor
        self.text_col = text_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.image_dir, row["IMAGE"])
        image = Image.open(img_path).convert("RGB")

        # image -> pixel values
        pixel_values = self.processor(
            images=image,
            return_tensors="pt",
        ).pixel_values.squeeze()

        # text -> token ids
        labels = self.processor.tokenizer(
            row[self.text_col],
            return_tensors="pt",
        ).input_ids.squeeze()

        return {"pixel_values": pixel_values, "labels": labels}


def make_collate_fn(processor):
    """Build a collate_fn that pads variable-length label sequences."""

    pad_id = processor.tokenizer.pad_token_id

    def collate_fn(batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])

        labels = torch.nn.utils.rnn.pad_sequence(
            [item["labels"] for item in batch],
            batch_first=True,
            padding_value=pad_id,
        )

        return {"pixel_values": pixel_values, "labels": labels}

    return collate_fn
