import os

import pytest

from app.generator import ExtractiveGenerator
from app.orchestrator import Config, Orchestrator
from app.rag import HashEmbedder, Retriever

ROOT = os.path.dirname(os.path.dirname(__file__))


class FakeNLU:
    """Returns scripted NLU results in order."""
    def __init__(self, *results):
        self.results = list(results)

    def predict(self, text):
        r = dict(self.results.pop(0))
        r.setdefault("tokens", text.split())
        return r


def nlu(intent, conf=0.95, **ents):
    return {"intent": intent, "confidence": conf, "entities": {k.upper(): v for k, v in ents.items()}}


def make(*results):
    cfg = Config(symptom_routing_path=os.path.join(ROOT, "kb", "symptom_routing.json"))
    ret = Retriever(os.path.join(ROOT, "kb", "faq.json"), HashEmbedder())
    return Orchestrator(FakeNLU(*results), ret, ExtractiveGenerator(), cfg)


def test_booking_slot_filling_then_request_noted_not_confirmed():
    o = make(nlu("book_appointment", doctor=["dr sara malik"]), nlu("out_of_scope", 0.3, date=["tomorrow"]),
             nlu("out_of_scope", 0.2, time=["5 pm"]))
    r1 = o.handle("s", "book dr sara malik")
    assert r1["action"]["type"] == "slot_filling" and "day" in r1["reply"].lower()
    r2 = o.handle("s", "tomorrow")
    assert "time" in r2["reply"].lower()
    r3 = o.handle("s", "5 pm")
    assert r3["action"] == {"type": "book_appointment", "status": "requested",
                            "slots": {"DOCTOR": "dr sara malik", "DATE": "tomorrow", "TIME": "5 pm"}}
    assert "noted" in r3["reply"].lower() and "confirm" in r3["reply"].lower()


def test_cancel_needs_identifier_and_reschedule_takes_new_slot():
    o = make(nlu("cancel_appointment"), nlu("cancel_appointment", 0.9, date=["friday"]),
             nlu("reschedule_appointment", 0.9, date=["friday", "monday"]))
    assert o.handle("a", "cancel my appointment")["action"]["type"] == "slot_filling"
    assert o.handle("a", "friday one")["action"]["type"] == "cancel_appointment"
    r = o.handle("b", "move friday to monday")
    assert r["action"]["slots"]["DATE"] == "monday"      # last date = the new one


def test_faq_answers_in_users_language_with_source():
    o = make(nlu("clinic_info", 0.9))
    r = o.handle("s", "کلینک کے اوقات کیا ہیں")
    assert r["lang"] == "ur" and r["source"] == "timings" and "9" in r["reply"]


def test_symptom_routing_red_flag_and_no_diagnosis():
    o = make(nlu("symptom_inquiry", 0.9, symptom=["chest pain"]), nlu("symptom_inquiry", 0.9, symptom=["fever"]),
             nlu("symptom_inquiry", 0.9))
    r = o.handle("s", "I have chest pain")
    assert r["action"]["red_flag"] is True and "emergency" in r["reply"].lower() and "diagnosis" in r["reply"].lower()
    r = o.handle("s2", "my child has fever")
    assert r["action"]["department"] == "pediatrics"
    assert o.handle("s3", "I feel unwell")["action"]["type"] == "symptom_generic"


def test_low_confidence_clarifies_then_hands_off():
    o = make(nlu("book_appointment", 0.2), nlu("book_appointment", 0.2))
    assert o.handle("s", "asdf")["action"]["type"] == "clarify"
    r = o.handle("s", "qwerty")
    assert r["handoff"] is True


def test_talk_to_human_out_of_scope_and_task_switch():
    o = make(nlu("talk_to_human"), nlu("out_of_scope"), nlu("book_appointment", 0.9, doctor=["dr x"]),
             nlu("cancel_appointment", 0.9, date=["friday"]))
    assert o.handle("s", "human please")["handoff"] is True
    assert o.handle("s", "tell me a joke")["action"]["type"] == "out_of_scope"
    assert o.handle("t", "book dr x")["action"]["type"] == "slot_filling"
    r = o.handle("t", "actually cancel friday")                # user switches task mid-flow
    assert r["action"]["type"] == "cancel_appointment"


def test_session_language_kept_for_ambiguous_short_replies():
    o = make(nlu("book_appointment", 0.9, department=["cardiology"]), nlu("book_appointment", 0.9, date=["kal"]))
    o.handle("s", "mujhe cardiology mein appointment chahiye")
    assert o.handle("s", "kal")["lang"] == "rur"


def test_faq_answer_has_front_desk_exit_and_it_can_be_disabled():
    o = make(nlu("clinic_info", 0.9), nlu("clinic_info", 0.9))
    assert "front desk" in o.handle("a", "what are the clinic timings")["reply"].lower()
    o.cfg.faq_followup = False
    assert "front desk" not in o.handle("b", "what are the clinic timings")["reply"].lower()


def test_low_margin_triggers_no_answer_instead_of_a_guess():
    o = make(nlu("clinic_info", 0.9), nlu("clinic_info", 0.9))
    r = o.handle("a", "what are the clinic timings")
    assert r["action"]["type"] == "faq" and "margin" in r["action"]
    o.cfg.retrieval_min_margin = 10.0                       # impossible margin -> always abstain
    r = o.handle("b", "what are the clinic timings")
    assert r["action"]["type"] == "no_answer" and "front desk" in r["reply"].lower() and r["source"] is None
