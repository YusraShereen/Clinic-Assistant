# NLU Specification

## Languages
| Code | Variety | Notes |
|---|---|---|
| `en` | English | |
| `ur` | Urdu (Arabic script) | |
| `rur` | Roman Urdu (Latin script) | Spelling is non-standard; expect variation (kal / kl, subah / subha) |

## Intents (8)
| Intent | Meaning |
|---|---|
| `book_appointment` | Wants to book / check slots |
| `cancel_appointment` | Wants to cancel a booking |
| `reschedule_appointment` | Wants to move / change a booking |
| `clinic_info` | Timings, location, phone, parking, doctor/department availability |
| `fees_query` | Fees, charges, insurance, payment methods |
| `symptom_inquiry` | Describes symptoms / asks which doctor or department to see (no diagnosis given) |
| `talk_to_human` | Requests a receptionist, agent, or direct doctor contact |
| `out_of_scope` | Anything unrelated to the clinic |

## Entities (BIO tagging)
| Type | Tag when... | Example |
|---|---|---|
| `DOCTOR` | A doctor is named, **including the title** | `Dr Sara Malik`, `ڈاکٹر بلال حسین` |
| `DEPARTMENT` | A department / speciality is named | `cardiology`, `جلدی امراض` |
| `DATE` | A day is stated: relative or named weekday | `tomorrow`, `kal`, `جمعہ`, `next Monday` |
| `TIME` | A clock time, **including the day-part word** | `5 pm`, `shaam 5 baje`, `شام 5 بجے` |
| `SYMPTOM` | A symptom is stated | `chest pain`, `bukhar`, `سر درد` |

### Annotation rules
1. Tag only what is **explicitly mentioned**. Do not infer entities.
2. Tokenisation is whitespace-based. An entity must cover whole tokens.
3. Do not tag vague phrases ("last night", "two days") unless they match the DATE/TIME definitions above. Be consistent.
4. Two symptoms in one message = two separate `SYMPTOM` spans.
5. The safety boundary: `symptom_inquiry` routes to "which department / see a doctor" guidance, **never to diagnosis or treatment advice**.

## Data
- `data/synthetic/`: template-generated. Templates are split **by template** (test templates never appear in train). Metrics here are optimistic.
- `data/handwritten_test.jsonl`: **real evaluation set, written by hand in natural phrasing.** Target >= 20 per language. This is what the README's headline numbers must come from.
