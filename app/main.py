"""FastAPI entrypoint.  Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000"""
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from app.generator import build_generator
from app.orchestrator import Config, Orchestrator
from app.rag import Retriever, build_embedder

log = logging.getLogger("assistant")
logging.basicConfig(level=logging.INFO, format="%(message)s")
LOG_TEXT = os.getenv("LOG_TEXT", "0") == "1"      # raw user text is NOT logged unless enabled
_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)


class Metrics:
    def __init__(self):
        self.requests, self.handoffs, self.errors = 0, 0, 0
        self.by_intent, self.by_lang = {}, {}
        self.lat_sum, self.lat_count = 0.0, 0
        self.buckets = {b: 0 for b in _BUCKETS}

    def observe(self, out):
        self.requests += 1
        self.handoffs += int(out["handoff"])
        self.by_intent[out["intent"]] = self.by_intent.get(out["intent"], 0) + 1
        self.by_lang[out["lang"]] = self.by_lang.get(out["lang"], 0) + 1
        s = out["latency_ms"] / 1000
        self.lat_sum += s
        self.lat_count += 1
        for b in _BUCKETS:
            if s <= b:
                self.buckets[b] += 1

    def render(self):
        L = ["# TYPE assistant_requests_total counter", f"assistant_requests_total {self.requests}",
             "# TYPE assistant_handoffs_total counter", f"assistant_handoffs_total {self.handoffs}",
             "# TYPE assistant_errors_total counter", f"assistant_errors_total {self.errors}",
             "# TYPE assistant_intent_total counter"]
        L += [f'assistant_intent_total{{intent="{k}"}} {v}' for k, v in sorted(self.by_intent.items())]
        L += ["# TYPE assistant_language_total counter"]
        L += [f'assistant_language_total{{lang="{k}"}} {v}' for k, v in sorted(self.by_lang.items())]
        L += ["# TYPE assistant_latency_seconds histogram"]
        L += [f'assistant_latency_seconds_bucket{{le="{b}"}} {c}' for b, c in self.buckets.items()]
        L += [f'assistant_latency_seconds_bucket{{le="+Inf"}} {self.lat_count}',
              f"assistant_latency_seconds_sum {self.lat_sum:.6f}",
              f"assistant_latency_seconds_count {self.lat_count}"]
        return "\n".join(L) + "\n"


class ChatIn(BaseModel):
    text: str
    session_id: str | None = None


class TextIn(BaseModel):
    text: str


def build_orchestrator():
    from app.nlu_service import NLUService
    nlu = NLUService(os.getenv("NLU_MODEL_DIR", "models/nlu"), os.getenv("LABEL_MAP", "data/label_map.json"),
                     device=os.getenv("DEVICE", "cpu"))
    embedder = build_embedder(os.getenv("EMBEDDER", "e5"), os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small"))
    retriever = Retriever(os.getenv("KB_PATH", "kb/faq.json"), embedder)
    return Orchestrator(nlu, retriever, build_generator(), Config())


def create_app(orchestrator=None):
    state = {"orch": orchestrator, "metrics": Metrics()}

    @asynccontextmanager
    async def lifespan(_app):
        if state["orch"] is None:
            state["orch"] = build_orchestrator()
        yield

    app = FastAPI(title="Bilingual clinic assistant", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        o = state["orch"]
        return {"status": "ok" if o else "loading", "embedder": getattr(getattr(o, "retriever", None), "embedder", None)
                and o.retriever.embedder.name, "generator": getattr(getattr(o, "generator", None), "name", None)}

    @app.post("/chat")
    def chat(body: ChatIn):
        sid = body.session_id or uuid.uuid4().hex
        try:
            out = state["orch"].handle(sid, body.text)
        except Exception:  # noqa: BLE001
            state["metrics"].errors += 1
            log.exception("chat failed")
            raise
        state["metrics"].observe(out)
        rec = {"ts": round(time.time(), 3), "session": sid[:8], "lang": out["lang"], "intent": out["intent"],
               "conf": out["confidence"], "handoff": out["handoff"], "ms": out["latency_ms"]}
        if LOG_TEXT:
            rec["text"] = body.text
        log.info(json.dumps(rec, ensure_ascii=False))
        out["session_id"] = sid
        return out

    @app.post("/nlu")
    def nlu(body: TextIn):
        return state["orch"].nlu.predict(body.text)

    @app.get("/metrics")
    def metrics():
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(state["metrics"].render())

    return app


app = create_app() if os.getenv("ASSISTANT_AUTOLOAD", "1") == "1" else None
