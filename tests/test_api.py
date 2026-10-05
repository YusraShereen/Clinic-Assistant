import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from fastapi.testclient import TestClient

from app.generator import ExtractiveGenerator, OpenAICompatGenerator
from app.main import create_app
from app.orchestrator import Config, Orchestrator
from app.rag import HashEmbedder, Retriever
from tests.test_orchestrator import FakeNLU, nlu

ROOT = os.path.dirname(os.path.dirname(__file__))


def _orch(*results, generator=None):
    cfg = Config(symptom_routing_path=os.path.join(ROOT, "kb", "symptom_routing.json"))
    return Orchestrator(FakeNLU(*results), Retriever(os.path.join(ROOT, "kb", "faq.json"), HashEmbedder()),
                        generator or ExtractiveGenerator(), cfg)


def test_health_chat_and_metrics():
    client = TestClient(create_app(_orch(nlu("clinic_info", 0.9), nlu("talk_to_human", 0.9))))
    assert client.get("/health").json()["status"] == "ok"
    r = client.post("/chat", json={"text": "what are the clinic timings", "session_id": "x"}).json()
    assert r["intent"] == "clinic_info" and r["session_id"] == "x" and r["source"] == "timings"
    assert client.post("/chat", json={"text": "agent please"}).json()["handoff"] is True
    m = client.get("/metrics").text
    assert "assistant_requests_total 2" in m and "assistant_handoffs_total 1" in m
    assert 'assistant_intent_total{intent="clinic_info"} 1' in m


def test_openai_compatible_generator_and_fallback():
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.headers["Authorization"] == "Bearer k"
            body = json.dumps({"choices": [{"message": {"content": "Open 9 to 9."}}]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    doc = json.load(open(os.path.join(ROOT, "kb", "faq.json"), encoding="utf-8"))[0]
    g = OpenAICompatGenerator(f"http://127.0.0.1:{srv.server_port}", "k", "m")
    assert g.answer(doc, "en", "when are you open") == "Open 9 to 9."
    srv.shutdown()
    dead = OpenAICompatGenerator("http://127.0.0.1:1", "k", "m", timeout=1)   # unreachable -> extractive fallback
    assert dead.answer(doc, "en", "q") == doc["en"]["a"]
