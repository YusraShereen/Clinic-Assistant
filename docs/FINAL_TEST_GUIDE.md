# How to write `data/final_test.jsonl` (the only set whose numbers you may quote)

## Why it matters
Everything else was used to build the model: synthetic data (templates), the dev set (its errors drove fixes).
The final test set is the one honest measurement of how the system handles messages it has never been tuned on.

## Rules
1. **Write it in your own words**, as real patients would message a clinic. Do not copy templates, dev examples, or my 15 format examples.
2. **Do not look at model predictions or error files while writing it.** Freeze the NLU first (see NEXT_STEPS step C).
3. **Evaluate on it once.** Whatever number comes out is the number you report. No re-tuning afterwards.
4. **Size:** at least **24 per language** (en / ur / rur) = 72+ total. More is better (40+/language gives much tighter numbers).
5. **Coverage:** at least **3 per intent per language** (8 intents x 3 = 24 per language).

## What to include (this is what real traffic looks like)
- Typos, shorthand, missing vowels, mixed casing, Roman Urdu spelling variants (kl/kal, shaam/sham, nahi/nai).
- **Code-mixing**: English words inside Urdu or Roman Urdu sentences.
- Short messages ("fees?") and long, rambling ones.
- Messages with 0, 1, 2 or 3 entities.
- Different ways to ask the same thing (do not reuse one phrasing).
- **Hard cases the model has struggled with**: payment-method questions, "what time" questions, day-parts
  ("evening", "shaam ko"), family members ("my mother", "ami ko"), body parts, past-time words ("since yesterday"),
  "which department for X?", requests for a human, completely unrelated questions (some that *sound* clinic-like).
- A few messages containing doctor names, departments or symptoms that are **not** in the data I generated.

## Labels: intents (pick exactly one)
`book_appointment`, `cancel_appointment`, `reschedule_appointment`, `clinic_info`, `fees_query`,
`symptom_inquiry`, `talk_to_human`, `out_of_scope`  (definitions: docs/SPEC.md)

## Labels: entities (tag only what is explicitly written)
| Type | Tag | Do NOT tag |
|---|---|---|
| DOCTOR | doctor name incl. title: `dr sara malik`, `ڈاکٹر بلال حسین` | "my doctor", "the doctor" |
| DEPARTMENT | department or role phrase: `cardiology`, `skin doctor`, `دل کے ڈاکٹر`, `gynae` | "clinic", "hospital" |
| DATE | day for an appointment: `tomorrow`, `kal`, `جمعہ`, `next week` | past-time ("yesterday" in "fever since yesterday"), "evening" alone |
| TIME | clock time incl. day-part word: `5 pm`, `shaam 5 baje`, `شام 6 بجے` | "evening" / "subah" alone, "kitne baje" |
| SYMPTOM | each symptom: `fever`, `bukhar`, `سر درد`, `backache` | body parts, family words, "pain" if it is part of a longer symptom you already tagged |

If a case is genuinely ambiguous, **write a different sentence** rather than guessing; do not leave debatable labels in the test set.

## File format (one JSON object per line; UTF-8)
```json
{"lang": "rur", "text": "mera slot parson 6 baje kar dein", "intent": "reschedule_appointment", "entities": [{"type": "DATE", "text": "parson"}, {"type": "TIME", "text": "6 baje"}]}
```
- `lang`: `en`, `ur` or `rur`.
- **Entity `text` must match whole space-separated words exactly as typed**, in order.
- **Write the text without punctuation attached to words** (no `tomorrow?`, `5 pm,`). The validator splits on spaces.
- Entities that appear twice (e.g. two times) are listed twice, in order of appearance.
- `"entities": []` for messages with none.

## 15 format examples (5 per language) - for FORMAT only, do not include them
See `docs/final_test_examples.jsonl`. They show: an availability question, a booking with a role phrase + date + time, a cancel with an untagged day-part, a symptom with an unseen symptom word, two symptoms in one sentence, an out-of-scope question, a payment question with no entities, a reschedule with date + time, and a nearby-place question.

## Checklist before you save
- [ ] >= 24 per language, >= 3 per intent per language
- [ ] Wrote everything myself, in my own phrasing
- [ ] Did not look at model errors while writing
- [ ] `python scripts/validate_handwritten.py` prints OK for `final_test` and shows overlap with synthetic train = 0
- [ ] Saved as `data/final_test.jsonl`, committed to git **before** running the final evaluation

## Same idea for retrieval: `data/rag_test.jsonl`
Write **at least 25 questions** (about 8 per language) a patient might ask whose answer is in `kb/faq.json`, phrased differently from the FAQ questions, plus the correct `doc_id`:
```json
{"lang": "ur", "question": "ٹیکے لگوانے ہیں بچے کے", "doc_id": "vaccinations"}
```
Include some that mix languages and some with typos. `scripts/eval_rag.py` validates the doc ids.
