# Doctor's Prescription Reader

Fine-tuned [TrOCR](https://huggingface.co/microsoft/trocr-base-handwritten) for
recognizing **handwritten medicine names** from doctor's prescriptions, with a
brand → generic name lookup and a Streamlit demo.

## Project structure

```
doctor-prescription-reader/
├── app/                  # Streamlit application
│   ├── app.py            # Main UI
│   ├── inference.py      # Loads model & predicts (cached)
│   ├── medicine_lookup.py# Brand → generic lookup
│   ├── image_processing.py
│   └── utils.py
├── models/
│   └── trocr_prescription_model/   # Fine-tuned weights (after training)
├── notebooks/
│   └── training.ipynb    # Colab training notebook
├── src/
│   ├── dataset.py        # PrescriptionDataset + collate_fn
│   ├── train.py          # Training script
│   ├── evaluate.py       # CER / WER / accuracy
│   ├── predict.py        # PrescriptionReader helper
│   ├── metrics.py        # Evaluation metrics
│   └── config.py         # Constants & paths
├── data/                 # Dataset instructions + sample images
├── assets/               # Logo, demo gif, screenshots
├── requirements.txt
├── setup.py
└── LICENSE
```

## Setup

```bash
pip install -r requirements.txt
pip install -e .          # so `src` and `app` are importable
```

## Train

```bash
python -m src.train --base_path "/path/to/Doctor's Handwritten Prescription BD dataset"
```

## Evaluate

```bash
python -m src.evaluate --base_path "/path/to/Doctor's Handwritten Prescription BD dataset"
```

## Run the app

```bash
streamlit run app/app.py
```

## Dataset

Doctor's Handwritten Prescription BD Dataset (Kaggle:
`mamun1113/doctors-handwritten-prescription-bd-dataset`) — 3120 train / 780 val
/ 780 test word-crop images. See [data/README.md](data/README.md).

## License

MIT — see [LICENSE](LICENSE).
