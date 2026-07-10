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
from src.detection import WordDetector, _to_rgb


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
        page = _to_rgb(page_image)
        regions = self.detector.detect(page)

        medicines = []
        for region in regions:
            result = reader.read(region["crop"])
            self._append_match(medicines, result, region["bbox"])

        if not medicines:
            # The upload may already be a single word crop (like the dataset
            # images), which the page detector over-segments into fragments.
            # Reading the whole image often recovers it.
            result = reader.read(page)
            self._append_match(medicines, result, (0, 0, *page.size))

        # top-to-bottom, then left-to-right
        medicines.sort(key=lambda m: (m["bbox"][1], m["bbox"][0]))
        return medicines

    def _append_match(self, medicines, result, bbox):
        """Match an OCR result against the medicine list; append if it hits."""
        text = result["text"]
        if not text:
            return
        match = match_medicine(text, cutoff=self.match_cutoff)
        if match is None:
            return  # not a recognized medicine -> drop
        medicines.append(
            {
                "bbox": tuple(int(v) for v in bbox),
                "raw_text": text,
                "matched_brand": match["matched_brand"],
                "generic": match["generic"],
                "ocr_confidence": round(result["confidence"], 3),
                "match_score": round(match["score"], 3),
            }
        )
