"""Full-page prescription pipeline: detect -> recognize -> filter -> lookup.

    page image
      -> WordDetector finds candidate word crops
      -> PrescriptionReader transcribes each crop (+ confidence)
      -> fuzzy-match each reading against the known medicine list
      -> keep matches, attach generic name

The recognizer is loaded lazily so the detection/preview path works even before
the fine-tuned model (and torch) are available.
"""

from app.medicine_lookup import match_medicine
from src.detection import WordDetector


class PrescriptionPipeline:
    def __init__(self, reader=None, detector_backend="auto", match_cutoff=0.7):
        self.detector = WordDetector(backend=detector_backend)
        self.match_cutoff = match_cutoff
        self._reader = reader  # may be None until the model exists

    # -- lazy model load ------------------------------------------------
    def _get_reader(self):
        if self._reader is None:
            from src.predict import PrescriptionReader  # heavy import

            self._reader = PrescriptionReader()
        return self._reader

    # -- stages ---------------------------------------------------------
    def detect(self, page_image):
        """Stage 1 only: return detected word regions (bbox + crop)."""
        return self.detector.detect(page_image)

    def read_page(self, page_image):
        """Full pipeline. Returns a list of medicine dicts, sorted by position.

        Each item: {bbox, raw_text, matched_brand, generic, ocr_confidence,
        match_score}.
        """
        reader = self._get_reader()
        regions = self.detector.detect(page_image)

        medicines = []
        for region in regions:
            result = reader.read(region["crop"])
            text = result["text"]
            if not text:
                continue

            match = match_medicine(text, cutoff=self.match_cutoff)
            if match is None:
                continue  # not a recognized medicine -> drop

            medicines.append(
                {
                    "bbox": region["bbox"],
                    "raw_text": text,
                    "matched_brand": match["matched_brand"],
                    "generic": match["generic"],
                    "ocr_confidence": round(result["confidence"], 3),
                    "match_score": round(match["score"], 3),
                }
            )

        # top-to-bottom, then left-to-right
        medicines.sort(key=lambda m: (m["bbox"][1], m["bbox"][0]))
        return medicines
