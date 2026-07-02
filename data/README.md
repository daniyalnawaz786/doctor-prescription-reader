# Data

This project uses the **Doctor's Handwritten Prescription BD Dataset** from Kaggle:
`mamun1113/doctors-handwritten-prescription-bd-dataset`

## Structure (after download)

```
Doctor's Handwritten Prescription BD dataset/
├── Training/
│   ├── training_labels.csv     # IMAGE, MEDICINE_NAME, GENERIC_NAME
│   └── training_words/         # 3120 word-crop PNGs
├── Validation/
│   ├── validation_labels.csv
│   └── validation_words/       # 780 PNGs
└── Testing/
    ├── testing_labels.csv
    └── testing_words/          # 780 PNGs
```

## Download

```python
import kagglehub
path = kagglehub.dataset_download("mamun1113/doctors-handwritten-prescription-bd-dataset")
```

`sample_images/` holds a few example crops for the demo app. The full dataset is
not committed to the repo.
