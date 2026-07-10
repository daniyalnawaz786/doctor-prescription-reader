"""Prediction helper: load the fine-tuned model and transcribe an image."""

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src import config


class PrescriptionReader:
    """Loads the fine-tuned TrOCR model once and predicts medicine names."""

    def __init__(self, model_dir=config.MODEL_DIR):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = TrOCRProcessor.from_pretrained(str(model_dir))
        self.model = VisionEncoderDecoderModel.from_pretrained(str(model_dir))
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def read(self, image):
        """Transcribe an image, returning {"text": str, "confidence": float}.

        Confidence is the model's mean per-token probability (0-1), derived from
        the generation scores — a decent proxy for how sure it is about the read.
        """
        if isinstance(image, (str, bytes)):
            image = Image.open(image)
        image = image.convert("RGB")

        pixel_values = self.processor(
            images=image,
            return_tensors="pt",
            # Tiny crops (e.g. 1x50) make channel inference guess wrong; we
            # always pass RGB PIL images, so the layout is known.
            input_data_format="channels_last",
        ).pixel_values.to(self.device)

        out = self.model.generate(
            pixel_values,
            max_length=config.MAX_TARGET_LENGTH,
            output_scores=True,
            return_dict_in_generate=True,
        )

        text = self.processor.batch_decode(
            out.sequences, skip_special_tokens=True
        )[0].strip()

        confidence = self._confidence(out)
        return {"text": text, "confidence": confidence}

    def predict(self, image):
        """Convenience: return just the decoded text."""
        return self.read(image)["text"]

    def _confidence(self, generate_output):
        """Mean per-token probability from generation scores, as a float 0-1."""
        try:
            kwargs = {}
            # Beam search misattributes scores unless beam_indices is passed.
            if getattr(generate_output, "beam_indices", None) is not None:
                kwargs["beam_indices"] = generate_output.beam_indices
            transition = self.model.compute_transition_scores(
                generate_output.sequences,
                generate_output.scores,
                normalize_logits=True,
                **kwargs,
            )
            # transition holds log-probs of each generated token (padding is -inf)
            logprobs = transition[0]
            finite = logprobs[torch.isfinite(logprobs)]
            if finite.numel() == 0:
                return 0.0
            return float(finite.mean().exp())
        except Exception:
            return 0.0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to a word-crop image")
    args = parser.parse_args()

    reader = PrescriptionReader()
    print("Prediction:", reader.predict(args.image))
