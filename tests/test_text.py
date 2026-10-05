from app.text import decode_entities, detect_lang, normalize


def test_normalize_strips_punctuation_but_keeps_time_colon():
    assert normalize("What time? Is 4:30 pm ok, please!") == ["What", "time", "Is", "4:30", "pm", "ok", "please"]
    assert normalize("کیا کلینک کھلا ہے؟") == ["کیا", "کلینک", "کھلا", "ہے"]
    assert normalize("") == [] and normalize(None) == []


def test_detect_lang():
    assert detect_lang("کل شام 5 بجے") == ("ur", True)
    assert detect_lang("mujhe kal appointment chahiye") == ("rur", True)
    assert detect_lang("I want to book an appointment") == ("en", True)
    assert detect_lang("5 pm")[1] is False
    assert detect_lang("hello")[1] is False


def test_decode_entities_bio_and_lenient_i():
    toks = "book dr sara malik tomorrow at 5 pm".split()
    tags = ["O", "B-DOCTOR", "I-DOCTOR", "I-DOCTOR", "B-DATE", "O", "B-TIME", "I-TIME"]
    assert decode_entities(toks, tags) == {"DOCTOR": ["dr sara malik"], "DATE": ["tomorrow"], "TIME": ["5 pm"]}
    # I- right after another type starts a new span (model quirk)
    assert decode_entities(["8", "pm", "tomorrow"], ["B-TIME", "I-TIME", "I-DATE"]) == {"TIME": ["8 pm"], "DATE": ["tomorrow"]}
    assert decode_entities(["a", "b"], ["O", "O"]) == {}


def test_decode_merges_adjacent_same_type_pieces():
    toks = "reschedule to next week please".split()
    tags = ["O", "O", "B-DATE", "B-DATE", "O"]                 # model quirk: 2nd word tagged B-
    assert decode_entities(toks, tags) == {"DATE": ["next week"]}
    assert decode_entities(toks, tags, merge_adjacent=False) == {"DATE": ["next", "week"]}
    # symptoms are NOT merged: two symptoms may be adjacent
    assert decode_entities(["bukhar", "khansi"], ["B-SYMPTOM", "B-SYMPTOM"]) == {"SYMPTOM": ["bukhar", "khansi"]}
