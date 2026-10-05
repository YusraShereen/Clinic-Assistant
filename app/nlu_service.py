"""Wraps the fine-tuned joint intent+NER model for serving."""
from pathlib import Path

from app.text import decode_entities, normalize


class NLUService:
    def __init__(self, model_dir, label_map_path="data/label_map.json", device="auto", max_len=64):
        import json
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from nlu.core import JointNLU

        self.torch = torch
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                                   else ("cpu" if device == "auto" else device))
        self.max_len = max_len
        self.lm = json.loads(Path(label_map_path).read_text(encoding="utf-8"))
        mdir = Path(model_dir)
        self.tok = AutoTokenizer.from_pretrained(mdir)
        enc = AutoModel.from_config(AutoConfig.from_pretrained(mdir))
        self.model = JointNLU(enc, len(self.lm["intents"]), len(self.lm["ner_labels"]))
        self.model.load_state_dict(torch.load(mdir / "model.pt", map_location="cpu"))
        self.model.to(self.device).eval()
        torch.set_num_threads(max(1, torch.get_num_threads()))

    def predict(self, text):
        from nlu.core import predict
        tokens = normalize(text)
        if not tokens:
            return {"intent": "out_of_scope", "confidence": 0.0, "entities": {}, "tokens": []}
        intents, tags, confs = predict(self.model, self.tok, [tokens], self.lm, self.device,
                                       self.max_len, bs=1, return_conf=True)
        return {"intent": intents[0], "confidence": float(confs[0]),
                "entities": decode_entities(tokens, tags[0]), "tokens": tokens}
