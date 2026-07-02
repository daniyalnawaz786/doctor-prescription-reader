"""Map a predicted brand (medicine) name to its generic name.

The lookup table is built from the dataset's brand -> generic pairs. If no
exact match is found, a fuzzy match is attempted so small OCR errors still
resolve to the right medicine.
"""

import difflib

# Minimal seed table; extend by loading from the dataset CSVs at startup.
BRAND_TO_GENERIC = {
    "Napa": "Paracetamol",
    "Tamen": "Paracetamol",
    "Napa Extend": "Paracetamol",
    "Axodin": "Fexofenadine Hydrochloride",
    "Telfast": "Fexofenadine Hydrochloride",
    "Montex": "Montelukast Sodium",
    "Trilock": "Montelukast Sodium",
    "Esoral": "Esomeprazole",
    "Opton": "Esomeprazole",
    "Disopan": "Clonazepam",
    "Tridosil": "Azithromycin Dihydrate",
}


def load_from_csv(csv_path, brand_col="MEDICINE_NAME", generic_col="GENERIC_NAME"):
    """Populate BRAND_TO_GENERIC from a labels CSV."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    BRAND_TO_GENERIC.update(
        dict(zip(df[brand_col].astype(str), df[generic_col].astype(str)))
    )
    return BRAND_TO_GENERIC


def lookup_generic(brand, cutoff=0.75):
    """Return (generic_name, matched_brand) or (None, None) if unresolved."""
    result = match_medicine(brand, cutoff=cutoff)
    if result is None:
        return None, None
    return result["generic"], result["matched_brand"]


def match_medicine(text, cutoff=0.75):
    """Fuzzy-match a piece of OCR text against the known brand list.

    Returns ``{"matched_brand", "generic", "score"}`` for the best match at or
    above ``cutoff``, else ``None`` (i.e. this text is probably not a medicine).
    """
    if not text:
        return None

    # exact hit (case-insensitive)
    for brand in BRAND_TO_GENERIC:
        if brand.lower() == text.lower():
            return {"matched_brand": brand, "generic": BRAND_TO_GENERIC[brand], "score": 1.0}

    best_brand, best_score = None, 0.0
    for brand in BRAND_TO_GENERIC:
        score = difflib.SequenceMatcher(None, text.lower(), brand.lower()).ratio()
        if score > best_score:
            best_brand, best_score = brand, score

    if best_brand is not None and best_score >= cutoff:
        return {
            "matched_brand": best_brand,
            "generic": BRAND_TO_GENERIC[best_brand],
            "score": best_score,
        }
    return None
