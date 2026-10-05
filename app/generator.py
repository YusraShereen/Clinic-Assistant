"""Answer generation. Default is extractive (returns the retrieved FAQ answer in the user's
language: zero cost, deterministic, cannot hallucinate). Optionally an OpenAI-compatible
chat endpoint (e.g. Groq) can rephrase the retrieved answer; any failure falls back to extractive."""
import json
import os
import urllib.request

_LANG_NAME = {"en": "English", "ur": "Urdu (Arabic script)", "rur": "Roman Urdu (Latin script)"}


class ExtractiveGenerator:
    name = "extractive"

    def answer(self, doc, lang, question):
        return doc[lang]["a"]


class OpenAICompatGenerator:
    name = "openai-compatible"

    def __init__(self, base_url, api_key, model, timeout=10, fallback=None):
        self.base_url = base_url.rstrip("/")
        self.api_key, self.model, self.timeout = api_key, model, timeout
        self.fallback = fallback or ExtractiveGenerator()

    def answer(self, doc, lang, question):
        context = f"{doc['en']['a']}\n{doc['ur']['a']}\n{doc['rur']['a']}"
        body = {
            "model": self.model, "temperature": 0, "max_tokens": 200,
            "messages": [
                {"role": "system", "content":
                    "You are a clinic front-desk assistant. Answer ONLY from the context. If the context "
                    "does not contain the answer, say you do not know. Never give medical advice. "
                    f"Reply in {_LANG_NAME[lang]}, in at most 3 short sentences."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        }
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                text = json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"].strip()
            return text or self.fallback.answer(doc, lang, question)
        except Exception:  # noqa: BLE001  (network/JSON/quota problems must never break the chat)
            return self.fallback.answer(doc, lang, question)


def build_generator():
    if os.getenv("GENERATOR", "extractive") == "openai":
        return OpenAICompatGenerator(os.environ["GENERATOR_URL"], os.environ["GENERATOR_KEY"],
                                     os.environ["GENERATOR_MODEL"])
    return ExtractiveGenerator()
