"""Dialogue orchestration: NLU -> routing -> (slot filling | FAQ retrieval | symptom routing | handoff).

IMPORTANT: this is a front-desk *assistant*, not a booking system. For booking / cancelling /
rescheduling it collects the details and records a REQUEST for the front desk to confirm; it never
claims an appointment is confirmed. It gives no diagnosis or medical advice.
"""
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.messages import MSG
from app.text import detect_lang

TRANSACTIONAL = {"book_appointment", "cancel_appointment", "reschedule_appointment"}
INFO_INTENTS = {"clinic_info", "fees_query", "symptom_inquiry", "out_of_scope"}
SLOT_KEYS = ("DOCTOR", "DEPARTMENT", "DATE", "TIME")
CHILD_WORDS = {"child", "son", "daughter", "baby", "kid", "bachay", "bachey", "bacha", "bachi", "beta", "beti",
               "بچے", "بچی", "بیٹے", "بیٹی", "بچہ"}


@dataclass
class Config:
    intent_threshold: float = field(default_factory=lambda: float(os.getenv("INTENT_THRESHOLD", "0.55")))
    retrieval_min_score: float = field(default_factory=lambda: float(os.getenv("RETRIEVAL_MIN_SCORE", "0.0")))
    # e5 scores do not separate answerable from unanswerable questions, but the GAP between the top-2
    # documents does (dev set: mean 0.031 vs 0.009). Below this margin the bot hands over instead of answering.
    retrieval_min_margin: float = field(default_factory=lambda: float(os.getenv("RETRIEVAL_MIN_MARGIN", "0.0")))
    max_low_conf: int = 2
    # Retrieval cannot reliably say "not in the knowledge base" (e5 scores do not separate answerable from
    # unanswerable questions), so every FAQ answer carries an exit to the front desk.
    faq_followup: bool = field(default_factory=lambda: os.getenv("FAQ_FOLLOWUP", "1") == "1")
    clinic_phone: str = field(default_factory=lambda: os.getenv("CLINIC_PHONE", "021-000-0000"))
    symptom_routing_path: str = field(default_factory=lambda: os.getenv("SYMPTOM_ROUTING", "kb/symptom_routing.json"))
    max_sessions: int = 2000


class Orchestrator:
    def __init__(self, nlu, retriever, generator, cfg=None):
        self.nlu, self.retriever, self.generator = nlu, retriever, generator
        self.cfg = cfg or Config()
        self.sessions = {}
        routing = json.loads(Path(self.cfg.symptom_routing_path).read_text(encoding="utf-8"))
        self.departments = routing["departments"]
        self._symptom_index = [(k.lower(), e) for e in routing["symptoms"] for k in e["keys"]]

    # ---------------------------------------------------------------- public
    def handle(self, session_id, text):
        t0 = time.perf_counter()
        sess = self._session(session_id)
        lang, confident = detect_lang(text)
        lang = lang if confident else (sess["lang"] or lang)
        sess["lang"] = lang
        nlu = self.nlu.predict(text)
        out = self._route(sess, lang, text, nlu["intent"], nlu["confidence"], nlu["entities"], nlu.get("tokens", []))
        out.update({"lang": lang, "intent": nlu["intent"], "confidence": round(nlu["confidence"], 4),
                    "entities": nlu["entities"],
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 2)})
        return out

    # --------------------------------------------------------------- helpers
    def _session(self, sid):
        if sid not in self.sessions:
            if len(self.sessions) >= self.cfg.max_sessions:      # crude eviction: drop the oldest
                self.sessions.pop(next(iter(self.sessions)))
            self.sessions[sid] = {"lang": None, "pending": None, "slots": {}, "low": 0}
        return self.sessions[sid]

    @staticmethod
    def _reply(lang, key, action=None, source=None, handoff=False, **fmt):
        return {"reply": MSG[lang][key].format(**fmt) if fmt else MSG[lang][key],
                "action": action, "source": source, "handoff": handoff}

    def _handoff(self, sess, lang):
        sess.update({"pending": None, "slots": {}, "low": 0})
        return self._reply(lang, "handoff", action={"type": "handoff"}, handoff=True, phone=self.cfg.clinic_phone)

    # --------------------------------------------------------------- routing
    def _route(self, sess, lang, text, intent, conf, ents, tokens):
        thr = self.cfg.intent_threshold
        pending = sess["pending"]
        has_slot_entity = any(k in ents for k in SLOT_KEYS)

        if pending:
            if intent == "talk_to_human" and conf >= thr:
                return self._handoff(sess, lang)
            if intent in TRANSACTIONAL and conf >= thr and intent != pending:
                sess.update({"pending": None, "slots": {}})        # user switched task
                return self._transact(sess, lang, intent, ents)
            if has_slot_entity:
                return self._transact(sess, lang, pending, ents)
            if conf >= thr and intent in INFO_INTENTS:             # side question; keep the task open
                return self._info(sess, lang, text, intent, ents, tokens)
            return self._low_conf(sess, lang)

        if conf < thr:
            return self._low_conf(sess, lang)
        sess["low"] = 0
        if intent == "talk_to_human":
            return self._handoff(sess, lang)
        if intent in TRANSACTIONAL:
            return self._transact(sess, lang, intent, ents)
        return self._info(sess, lang, text, intent, ents, tokens)

    def _low_conf(self, sess, lang):
        sess["low"] += 1
        if sess["low"] >= self.cfg.max_low_conf:
            return self._handoff(sess, lang)
        return self._reply(lang, "low_conf", action={"type": "clarify"})

    def _info(self, sess, lang, text, intent, ents, tokens):
        if intent == "out_of_scope":
            return self._reply(lang, "oos", action={"type": "out_of_scope"})
        if intent == "symptom_inquiry":
            return self._symptom(lang, ents, tokens)
        query = text
        for k in ("DOCTOR", "DEPARTMENT"):
            if k in ents:
                query += " " + " ".join(ents[k])
        hits = self.retriever.search(query, k=3)
        margin = hits[0]["score"] - hits[1]["score"] if len(hits) > 1 else 1.0
        if (not hits or hits[0]["score"] < self.cfg.retrieval_min_score
                or margin < self.cfg.retrieval_min_margin):
            return self._reply(lang, "no_answer", action={"type": "no_answer", "margin": round(margin, 4)})
        doc = self.retriever.get(hits[0]["doc_id"])
        answer = self.generator.answer(doc, lang, text)
        if self.cfg.faq_followup:
            answer = answer + " " + MSG[lang]["faq_followup"]
        return {"reply": answer, "action": {"type": "faq", "retrieval_score": round(hits[0]["score"], 4), "margin": round(margin, 4)},
                "source": hits[0]["doc_id"], "handoff": False}

    def _symptom(self, lang, ents, tokens):
        found = None
        for s in ents.get("SYMPTOM", []):
            low = s.lower()
            for key, entry in self._symptom_index:
                if key == low or key in low:
                    found = (s, entry)
                    break
            if found:
                break
        if not found:
            return self._reply(lang, "symptom_generic", action={"type": "symptom_generic"})
        symptom, entry = found
        dept = "pediatrics" if any(w.lower() in CHILD_WORDS for w in tokens) else entry["department"]
        out = self._reply(lang, "symptom_dept", action={"type": "symptom_routing", "department": dept,
                                                       "red_flag": bool(entry.get("red_flag"))},
                          symptom=symptom, department=self.departments[dept][lang])
        if entry.get("red_flag"):
            out["reply"] += " " + MSG[lang]["emergency"]
        return out

    def _transact(self, sess, lang, kind, ents):
        slots = sess["slots"]
        for key in SLOT_KEYS:
            vals = ents.get(key)
            if vals:
                slots[key] = vals[-1] if (kind == "reschedule_appointment" and key in ("DATE", "TIME")) else vals[0]
        missing = None
        if kind == "book_appointment":
            if "DOCTOR" not in slots and "DEPARTMENT" not in slots:
                missing = "ask_target"
            elif "DATE" not in slots:
                missing = "ask_date"
            elif "TIME" not in slots:
                missing = "ask_time"
        elif kind == "cancel_appointment":
            if not any(k in slots for k in ("DOCTOR", "DATE", "TIME")):
                missing = "ask_which_appt"
        elif kind == "reschedule_appointment":
            if "DATE" not in slots and "TIME" not in slots:
                missing = "ask_new_slot"
        if missing:
            sess["pending"] = kind
            return self._reply(lang, missing, action={"type": "slot_filling", "task": kind, "slots": dict(slots)})
        summary = ", ".join(slots[k] for k in SLOT_KEYS if k in slots)
        done = dict(slots)
        sess.update({"pending": None, "slots": {}, "low": 0})
        return self._reply(lang, "noted", action={"type": kind, "status": "requested", "slots": done},
                           kind=MSG[lang]["kind"][kind], summary=summary)
