#!/usr/bin/env python3
"""
Synthetic data generator for the bilingual clinic assistant.

Produces intent + NER (BIO) data in three language varieties:
  en  = English
  ur  = Urdu (Arabic script)
  rur = Roman Urdu (Latin script)

IMPORTANT (honesty note for the README):
  This data is SYNTHETIC (template + slot filling). Metrics on it are optimistic.
  The template-level split below reduces leakage (test templates are never seen
  in training), but the *real* evaluation must use data/handwritten_test.jsonl.

Usage:
  python scripts/generate_data.py
  python scripts/generate_data.py --train-per-template 20 --seed 7
"""
import argparse
import json
import random
import re
from collections import Counter
from itertools import product
from pathlib import Path

LANGS = ["en", "ur", "rur"]
ENTITY_TYPES = ["DOCTOR", "DEPARTMENT", "DATE", "TIME", "SYMPTOM"]

# --------------------------------------------------------------------------
# SLOT VALUES  (whitespace-tokenised; multi-token values get B-/I- tags)
# --------------------------------------------------------------------------
SLOTS = {
    "DOCTOR": {
        "en": ["Dr Ahmed Khan", "Dr Sara Malik", "Dr Bilal Hussain",
               "Dr Ayesha Siddiqui", "Dr Imran Sheikh", "Dr Fatima Noor"],
        "ur": ["ڈاکٹر احمد خان", "ڈاکٹر سارہ ملک", "ڈاکٹر بلال حسین",
               "ڈاکٹر عائشہ صدیقی", "ڈاکٹر عمران شیخ", "ڈاکٹر فاطمہ نور"],
        "rur": ["Dr Ahmed Khan", "Doctor Sara Malik", "Dr Bilal Hussain",
                "Doctor Ayesha Siddiqui", "Dr Imran Sheikh", "Dr Fatima Noor"],
    },
    "DEPARTMENT": {
        "en": ["cardiology", "dermatology", "pediatrics", "gynecology",
               "orthopedics", "ENT", "general medicine", "dentistry"],
        "ur": ["امراض قلب", "جلدی امراض", "بچوں کے امراض", "گائنی",
               "ہڈیوں کے امراض", "ای این ٹی", "جنرل میڈیسن", "دندان سازی"],
        "rur": ["cardiology", "skin", "bachon ke amraz", "gynae",
                "orthopedic", "ENT", "general medicine", "dentist"],
    },
    "DATE": {
        "en": ["tomorrow", "today", "Friday", "Saturday", "next Monday",
               "next week", "the day after tomorrow"],
        "ur": ["کل", "آج", "جمعہ", "ہفتہ", "اگلے پیر", "اگلے ہفتے", "پرسوں"],
        "rur": ["kal", "aaj", "juma", "hafta", "agle peer", "agle hafte", "parso"],
    },
    "TIME": {
        "en": ["5 pm", "10 am", "3 pm", "4:30 pm", "6 pm", "11:30 am"],
        "ur": ["شام 5 بجے", "صبح 10 بجے", "دوپہر 3 بجے", "شام 4:30 بجے",
               "شام 6 بجے", "صبح 11:30 بجے"],
        "rur": ["shaam 5 baje", "subah 10 baje", "dopahar 3 baje",
                "shaam 4:30 baje", "shaam 6 baje", "subah 11:30 baje"],
    },
    "SYMPTOM": {
        "en": ["fever", "headache", "chest pain", "cough", "stomach pain",
               "skin rash", "back pain", "sore throat"],
        "ur": ["بخار", "سر درد", "سینے میں درد", "کھانسی", "پیٹ درد",
               "جلد پر دانے", "کمر درد", "گلے میں درد"],
        "rur": ["bukhar", "sar dard", "seene mein dard", "khansi", "pait dard",
                "skin pe daane", "kamar dard", "gale mein dard"],
    },
}

# Optional conversational prefixes (tagged O). Adds surface variety.
PREFIXES = {
    "en": ["hi", "hello", "excuse me", "please"],
    "ur": ["السلام علیکم", "ہیلو", "سنیے"],
    "rur": ["salam", "hello", "suniye", "assalam o alaikum"],
}
PREFIX_PROB = 0.30

# Optional polite suffixes (tagged O).
SUFFIXES = {
    "en": ["please", "thanks", "thank you"],
    "ur": ["شکریہ", "مہربانی"],
    "rur": ["shukriya", "please", "thanks"],
}
SUFFIX_PROB = 0.20

# --------------------------------------------------------------------------
# TEMPLATES  (slots are exact whitespace-separated tokens like {DOCTOR};
#             do NOT attach punctuation to a slot)
# --------------------------------------------------------------------------
TEMPLATES = {
    "book_appointment": {
        "en": [
            "I want to book an appointment with {DOCTOR}",
            "book an appointment in {DEPARTMENT} for {DATE}",
            "can I see {DOCTOR} {DATE} at {TIME}",
            "I need to see someone in {DEPARTMENT}",
            "please schedule me with {DOCTOR} for {DATE}",
            "any slot available in {DEPARTMENT} {DATE}",
            "I would like to make a booking for {DATE} at {TIME}",
            "book me a slot",
            "I want to see a doctor",
        ],
        "ur": [
            "مجھے {DOCTOR} کے ساتھ اپوائنٹمنٹ بک کرنی ہے",
            "{DEPARTMENT} میں {DATE} کے لیے اپوائنٹمنٹ چاہیے",
            "کیا میں {DATE} {TIME} {DOCTOR} سے مل سکتا ہوں",
            "براہ کرم مجھے {DATE} کے لیے {DOCTOR} کے پاس وقت دیں",
            "مجھے {DEPARTMENT} کے ڈاکٹر سے ملنا ہے",
            "کیا {DEPARTMENT} میں {DATE} کوئی سلاٹ خالی ہے",
            "میری اپوائنٹمنٹ {TIME} کے لیے بک کر دیں",
            "مجھے ڈاکٹر کو دکھانا ہے",
            "ایک اپوائنٹمنٹ بک کر دیں",
        ],
        "rur": [
            "mujhe {DOCTOR} ke saath appointment book karni hai",
            "{DEPARTMENT} mein {DATE} ke liye appointment chahiye",
            "kya main {DATE} {TIME} {DOCTOR} se mil sakta hoon",
            "please mujhe {DATE} ke liye {DOCTOR} ke paas time dein",
            "mujhe {DEPARTMENT} ke doctor se milna hai",
            "kya {DEPARTMENT} mein {DATE} koi slot khali hai",
            "meri appointment {TIME} ke liye book kar dein",
            "mujhe doctor ko dikhana hai",
            "ek appointment book kar dein",
        ],
    },
    "cancel_appointment": {
        "en": [
            "cancel my appointment",
            "please cancel my appointment with {DOCTOR}",
            "cancel my appointment for {DATE}",
            "cancel the {TIME} booking",
            "I can't make it {DATE} so please cancel",
            "remove my booking with {DOCTOR}",
            "please cancel my {DEPARTMENT} appointment",
            "I no longer need the appointment",
            "delete my appointment",
        ],
        "ur": [
            "میری اپوائنٹمنٹ کینسل کر دیں",
            "براہ کرم {DOCTOR} کے ساتھ میری اپوائنٹمنٹ کینسل کریں",
            "{DATE} کی میری اپوائنٹمنٹ منسوخ کر دیں",
            "{TIME} والی بکنگ کینسل کر دیں",
            "میں {DATE} نہیں آ سکتا اس لیے کینسل کر دیں",
            "{DOCTOR} کے ساتھ میری بکنگ ختم کر دیں",
            "میری {DEPARTMENT} کی اپوائنٹمنٹ کینسل کریں",
            "مجھے اب اپوائنٹمنٹ کی ضرورت نہیں",
            "میری بکنگ ہٹا دیں",
        ],
        "rur": [
            "meri appointment cancel kar dein",
            "please {DOCTOR} ke saath meri appointment cancel karein",
            "{DATE} ki meri appointment cancel kar dein",
            "{TIME} wali booking cancel kar dein",
            "main {DATE} nahi aa sakta isliye cancel kar dein",
            "{DOCTOR} ke saath meri booking khatam kar dein",
            "meri {DEPARTMENT} ki appointment cancel karein",
            "mujhe ab appointment ki zaroorat nahi",
            "meri booking hata dein",
        ],
    },
    "reschedule_appointment": {
        "en": [
            "reschedule my appointment to {DATE}",
            "can I move my appointment to {TIME}",
            "I need to change my appointment with {DOCTOR}",
            "please shift my {DEPARTMENT} appointment to {DATE}",
            "move my booking to {DATE} at {TIME}",
            "can we change the time to {TIME}",
            "I want a different day for my appointment",
            "is it possible to postpone my appointment to {DATE}",
            "change my appointment timing",
        ],
        "ur": [
            "میری اپوائنٹمنٹ {DATE} پر منتقل کر دیں",
            "اپوائنٹمنٹ کا وقت {TIME} کر دیں",
            "مجھے {DOCTOR} کے ساتھ اپنی اپوائنٹمنٹ تبدیل کرنی ہے",
            "براہ کرم میری {DEPARTMENT} کی اپوائنٹمنٹ {DATE} کر دیں",
            "میری بکنگ {DATE} {TIME} کر دیں",
            "کیا وقت بدل کر {TIME} ہو سکتا ہے",
            "مجھے اپوائنٹمنٹ کے لیے کوئی اور دن چاہیے",
            "کیا اپوائنٹمنٹ {DATE} تک آگے ہو سکتی ہے",
            "میری اپوائنٹمنٹ کا وقت بدل دیں",
        ],
        "rur": [
            "meri appointment {DATE} par shift kar dein",
            "appointment ka waqt {TIME} kar dein",
            "mujhe {DOCTOR} ke saath apni appointment tabdeel karni hai",
            "please meri {DEPARTMENT} ki appointment {DATE} kar dein",
            "meri booking {DATE} {TIME} kar dein",
            "kya waqt badal kar {TIME} ho sakta hai",
            "mujhe appointment ke liye koi aur din chahiye",
            "kya appointment {DATE} tak aage ho sakti hai",
            "meri appointment ka time badal dein",
        ],
    },
    "clinic_info": {
        "en": [
            "what are your clinic timings",
            "what time is {DOCTOR} available",
            "when does the {DEPARTMENT} department open",
            "where is the clinic located",
            "is the clinic open {DATE}",
            "what is your address",
            "do you have parking",
            "do you have a {DEPARTMENT} department",
            "what is your phone number",
        ],
        "ur": [
            "آپ کے کلینک کے اوقات کیا ہیں",
            "{DOCTOR} کب دستیاب ہوتے ہیں",
            "{DEPARTMENT} کا شعبہ کب کھلتا ہے",
            "کلینک کہاں واقع ہے",
            "کیا کلینک {DATE} کھلا ہے",
            "آپ کا پتہ کیا ہے",
            "کیا آپ کے پاس پارکنگ ہے",
            "کیا آپ کے ہاں {DEPARTMENT} کا شعبہ ہے",
            "آپ کا فون نمبر کیا ہے",
        ],
        "rur": [
            "aap ke clinic ki timings kya hain",
            "{DOCTOR} kab available hote hain",
            "{DEPARTMENT} ka department kab khulta hai",
            "clinic kahan hai",
            "kya clinic {DATE} khula hai",
            "aap ka address kya hai",
            "kya aap ke paas parking hai",
            "kya aap ke yahan {DEPARTMENT} ka department hai",
            "aap ka phone number kya hai",
        ],
    },
    "fees_query": {
        "en": [
            "how much is the consultation fee",
            "what does {DOCTOR} charge",
            "what is the fee for {DEPARTMENT}",
            "what are the charges for a {DEPARTMENT} visit",
            "do you accept insurance",
            "is there a discount for follow-up visits",
            "tell me the price of a checkup",
            "how much do I need to pay for {DOCTOR}",
            "what payment methods do you accept",
        ],
        "ur": [
            "کنسلٹیشن فیس کتنی ہے",
            "{DOCTOR} کی فیس کتنی ہے",
            "{DEPARTMENT} کی فیس کیا ہے",
            "{DEPARTMENT} کے وزٹ کے چارجز کیا ہیں",
            "کیا آپ انشورنس قبول کرتے ہیں",
            "کیا فالو اپ وزٹ پر رعایت ہے",
            "چیک اپ کی قیمت بتائیں",
            "{DOCTOR} کو دکھانے کے لیے کتنے پیسے دینے ہوں گے",
            "آپ ادائیگی کے کون سے طریقے قبول کرتے ہیں",
        ],
        "rur": [
            "consultation fee kitni hai",
            "{DOCTOR} ki fee kitni hai",
            "{DEPARTMENT} ki fee kya hai",
            "{DEPARTMENT} visit ke charges kya hain",
            "kya aap insurance accept karte hain",
            "kya follow up visit par discount hai",
            "checkup ki qeemat bataein",
            "{DOCTOR} ko dikhane ke liye kitne paise dene honge",
            "aap payment ke kaun se tareeqe qabool karte hain",
        ],
    },
    "symptom_inquiry": {
        "en": [
            "I have had {SYMPTOM} for two days",
            "I am suffering from {SYMPTOM} which doctor should I see",
            "my child has {SYMPTOM}",
            "I have {SYMPTOM} and {SYMPTOM}",
            "what should I do about {SYMPTOM}",
            "is {SYMPTOM} serious",
            "I am having {SYMPTOM} since morning",
            "which department should I visit for {SYMPTOM}",
            "I am worried about my {SYMPTOM}",
        ],
        "ur": [
            "مجھے دو دن سے {SYMPTOM} ہے",
            "مجھے {SYMPTOM} ہے کس ڈاکٹر کو دکھاؤں",
            "میرے بچے کو {SYMPTOM} ہے",
            "مجھے {SYMPTOM} اور {SYMPTOM} ہے",
            "{SYMPTOM} کے لیے میں کیا کروں",
            "کیا {SYMPTOM} خطرناک ہے",
            "صبح سے مجھے {SYMPTOM} ہے",
            "{SYMPTOM} کے لیے کون سے شعبے میں جاؤں",
            "مجھے اپنے {SYMPTOM} کی فکر ہے",
        ],
        "rur": [
            "mujhe do din se {SYMPTOM} hai",
            "mujhe {SYMPTOM} hai kis doctor ko dikhaun",
            "mere bachay ko {SYMPTOM} hai",
            "mujhe {SYMPTOM} aur {SYMPTOM} hai",
            "{SYMPTOM} ke liye main kya karun",
            "kya {SYMPTOM} khatarnak hai",
            "subah se mujhe {SYMPTOM} hai",
            "{SYMPTOM} ke liye kis department mein jaun",
            "mujhe apne {SYMPTOM} ki fikar hai",
        ],
    },
    "talk_to_human": {
        "en": [
            "I want to talk to a human",
            "connect me to a receptionist",
            "can I speak to someone from the staff",
            "let me talk to an agent",
            "transfer me to a real person",
            "I need to speak to a doctor directly",
            "this is not helping please connect me to staff",
            "can someone call me back",
            "connect me to {DOCTOR} directly",
        ],
        "ur": [
            "میں کسی انسان سے بات کرنا چاہتا ہوں",
            "مجھے ریسپشنسٹ سے ملا دیں",
            "کیا میں عملے کے کسی فرد سے بات کر سکتا ہوں",
            "مجھے ایجنٹ سے بات کرنی ہے",
            "مجھے کسی حقیقی شخص سے ملائیں",
            "مجھے ڈاکٹر سے براہ راست بات کرنی ہے",
            "یہ مددگار نہیں ہے براہ کرم عملے سے ملائیں",
            "کیا کوئی مجھے واپس کال کر سکتا ہے",
            "مجھے {DOCTOR} سے براہ راست ملا دیں",
        ],
        "rur": [
            "main kisi insaan se baat karna chahta hoon",
            "mujhe receptionist se milwa dein",
            "kya main staff ke kisi bande se baat kar sakta hoon",
            "mujhe agent se baat karni hai",
            "mujhe kisi asli insaan se milwaein",
            "mujhe doctor se direct baat karni hai",
            "ye madad nahi kar raha please staff se milwa dein",
            "kya koi mujhe wapas call kar sakta hai",
            "mujhe {DOCTOR} se direct milwa dein",
        ],
    },
    "out_of_scope": {
        "en": [
            "what is the weather like",
            "tell me a joke",
            "who won the cricket match",
            "how do I make biryani",
            "what is the capital of France",
            "can you help me with my homework",
            "what is the dollar rate",
            "play some music",
            "recommend a good restaurant nearby",
        ],
        "ur": [
            "موسم کیسا ہے",
            "مجھے لطیفہ سنائیں",
            "کرکٹ میچ کون جیتا",
            "بریانی کیسے بنتی ہے",
            "فرانس کا دارالحکومت کیا ہے",
            "کیا آپ میرے ہوم ورک میں مدد کر سکتے ہیں",
            "ڈالر کا ریٹ کیا ہے",
            "کچھ موسیقی چلائیں",
            "قریب کوئی اچھا ریسٹورنٹ بتائیں",
        ],
        "rur": [
            "mausam kaisa hai",
            "mujhe latifa sunao",
            "cricket match kaun jeeta",
            "biryani kaise banti hai",
            "France ka capital kya hai",
            "kya aap meri homework mein madad kar sakte hain",
            "dollar ka rate kya hai",
            "kuch music chalao",
            "qareeb koi acha restaurant batao",
        ],
    },
}


# --------------------------------------------------------------------------
# v2 ADDITIONS  (from dev-set error analysis)
#  - weekday slots used with "on"/"ko" so prepositions are not tagged as DATE
#  - role phrases ("heart doctor") tagged as DEPARTMENT
#  - hard negatives: payment methods, body parts, family words, non-clinic price
#    questions -> tagged O
#  - availability-style clinic_info, "another day" reschedules, symptom+booking
#  - shorthand / casing variation
# Training words were chosen to be DIFFERENT from the words seen in dev errors
# (e.g. other payment methods, other spellings) so dev scores stay meaningful.
# --------------------------------------------------------------------------
SLOT_ENTITY = {  # slot name -> entity type (None = tagged O)
    "DATEWD": "DATE", "DEPTROLE": "DEPARTMENT", "LOCSYM": "SYMPTOM",
    "BODY": None, "FAMILY": None, "PAY": None, "PRICEITEM": None,
}

SLOTS.update({
    "DATEWD": {
        "en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        "ur": ["پیر", "منگل", "بدھ", "جمعرات", "جمعہ", "ہفتے", "اتوار"],
        "rur": ["peer", "mangal", "budh", "jumerat", "juma", "hafta", "itwaar",
                "monday", "friday", "sunday"],
    },
    "DEPTROLE": {
        "en": ["heart doctor", "child specialist", "bone doctor", "eye doctor",
               "cardiologist", "pediatrician", "dermatologist", "gynecologist"],
        "ur": ["دل کے ڈاکٹر", "بچوں کے ڈاکٹر", "ہڈیوں کے ڈاکٹر", "آنکھوں کے ڈاکٹر",
               "ماہر امراض قلب"],
        "rur": ["heart specialist", "bachon ke doctor", "haddiyon ke doctor",
                "aankhon ke doctor", "cardiologist", "gynae doctor"],
    },
    "LOCSYM": {
        "en": ["swelling", "itching", "pain", "redness", "stiffness"],
        "ur": ["سوجن", "خارش", "درد", "لالی", "اکڑن"],
        "rur": ["soojan", "kharish", "dard", "laali", "akrahat"],
    },
    "BODY": {
        "en": ["arm", "leg", "knee", "shoulder", "neck", "ankle"],
        "ur": ["بازو", "ٹانگ", "گھٹنے", "کندھے", "گردن", "ٹخنے"],
        "rur": ["bazu", "taang", "ghutna", "kandha", "gardan", "takhna"],
    },
    "FAMILY": {
        "en": ["mother", "father", "son", "daughter", "wife", "husband", "brother", "sister"],
        "ur": ["ابو", "امی", "بیٹے", "بیٹی", "بھائی", "بہن"],
        "rur": ["abu", "ami", "beta", "beti", "bhai", "behan"],
    },
    "PAY": {
        "en": ["cash", "debit card", "credit card", "bank transfer", "sadapay", "raast"],
        "ur": ["نقد", "ڈیبٹ کارڈ", "کریڈٹ کارڈ", "بینک ٹرانسفر", "ساداپے", "راست"],
        "rur": ["cash", "debit card", "credit card", "bank transfer", "sadapay", "raast"],
    },
    "PRICEITEM": {
        "en": ["gold", "silver", "diesel", "sugar", "flour", "tomatoes"],
        "ur": ["سونے", "چاندی", "ڈیزل", "چینی", "آٹے", "ٹماٹر"],
        "rur": ["gold", "chandi", "diesel", "cheeni", "atta", "tamatar"],
    },
})
# extra values for existing slots (shorthand / new surface forms)
SLOTS["DATE"]["en"] += ["tmrw", "tmr"]
SLOTS["DATE"]["rur"] += ["parson", "parsoon", "aj"]
SLOTS["SYMPTOM"]["en"] += ["swelling", "itching", "dizziness", "nausea", "weakness", "acidity"]
SLOTS["SYMPTOM"]["ur"] += ["سوجن", "خارش", "چکر", "متلی", "کمزوری", "تیزابیت"]
SLOTS["SYMPTOM"]["rur"] += ["soojan", "kharish", "chakkar", "matli", "kamzori", "acidity"]
PREFIXES["en"] += ["pls", "plz"]
PREFIXES["rur"] += ["pls", "bhai"]
SUFFIXES["en"] += ["plz"]
LOWER_PROB = 0.30   # lowercase the whole utterance (ent, dr, ...)

EXTRA_TEMPLATES = {
    "book_appointment": {
        "en": [
            "I want to book {DOCTOR} for my {FAMILY}",
            "can you fit my {FAMILY} in with {DEPARTMENT} on {DATEWD}",
            "I would like to see {DOCTOR} on {DATEWD}",
            "I want to see the {DEPTROLE}",
            "book me with the {DEPTROLE} for {DATE}",
            "I have {SYMPTOM} please book an appointment",
            "I wanted to book with {DOCTOR} on {DATEWD} at {TIME}",
            "any appointment in {DEPARTMENT} on {DATEWD}",
        ],
        "ur": [
            "{FAMILY} کے لیے {DOCTOR} کا وقت چاہیے",
            "{DEPARTMENT} میں {DATEWD} کو اپوائنٹمنٹ مل سکتی ہے",
            "مجھے {DATEWD} کو {DOCTOR} سے ملنا ہے",
            "مجھے {DEPTROLE} سے ملنا ہے",
            "{DEPTROLE} کا وقت چاہیے {DATE} کے لیے",
            "مجھے {SYMPTOM} ہے اپوائنٹمنٹ بک کر دیں",
            "{DEPARTMENT} کی بکنگ کرنی تھی {DATE}",
            "{DATEWD} کو {TIME} کا کوئی سلاٹ ہے",
        ],
        "rur": [
            "{FAMILY} ke liye {DOCTOR} ka time chahiye",
            "{DEPARTMENT} mein {DATEWD} ko appointment mil sakti hai",
            "mujhe {DATEWD} ko {DOCTOR} se milna hai",
            "mujhe {DEPTROLE} se milna hai",
            "{DEPTROLE} ka time chahiye {DATE} ke liye",
            "mujhe {SYMPTOM} hai appointment book kar dein",
            "{DEPARTMENT} ki booking karni thi {DATE}",
            "{DATEWD} ko {TIME} ka koi slot hai",
        ],
    },
    "cancel_appointment": {
        "en": [
            "I wont be able to make it on {DATEWD} please cancel",
            "cancel my {DATEWD} booking with {DOCTOR}",
            "cancel the booking for my {FAMILY}",
        ],
        "ur": [
            "{DATEWD} کی بکنگ کینسل کر دیں",
            "{FAMILY} کی اپوائنٹمنٹ کینسل کر دیں",
        ],
        "rur": [
            "{DATEWD} ki booking cancel kar dein",
            "{FAMILY} ki appointment cancel kar dein",
        ],
    },
    "reschedule_appointment": {
        "en": [
            "can we pick another day",
            "I cant come on {DATEWD} can we pick another day",
            "move it from {DATEWD} to {DATEWD}",
            "I need a different day with {DOCTOR}",
        ],
        "ur": [
            "مجھے کوئی دوسرا دن دے دیں",
            "{DATEWD} کو نہیں آ سکتا کوئی اور دن دے دیں",
            "{DATEWD} سے {DATEWD} پر منتقل کر دیں",
            "{DOCTOR} کے ساتھ کوئی اور دن چاہیے",
        ],
        "rur": [
            "koi dosra din de dein",
            "{DATEWD} ko nahi aa sakta koi aur din de dein",
            "{DATEWD} se {DATEWD} par shift kar dein",
            "{DOCTOR} ke saath koi aur din chahiye",
        ],
    },
    "clinic_info": {
        "en": [
            "is the clinic open on {DATEWD}",
            "do you work on {DATEWD}",
            "is {DOCTOR} in the clinic on {DATEWD}",
            "which days is {DOCTOR} available",
            "are you open in the evening",
            "until what time are you open",
            "is {DEPARTMENT} open on {DATEWD}",
            "is {DOCTOR} usually there in the mornings",
        ],
        "ur": [
            "کیا کلینک {DATEWD} کو کھلا ہے",
            "کیا آپ {DATEWD} کو کام کرتے ہیں",
            "کیا {DOCTOR} {DATEWD} کو کلینک میں ہوتے ہیں",
            "{DOCTOR} کون سے دن دستیاب ہوتے ہیں",
            "کیا آپ شام کو کھلے ہوتے ہیں",
            "آپ کتنے بجے تک کھلے ہیں",
            "کیا {DEPARTMENT} کا شعبہ {DATEWD} کو کھلا ہے",
            "کیا {DOCTOR} صبح کے وقت ہوتے ہیں",
        ],
        "rur": [
            "kya clinic {DATEWD} ko khula hai",
            "kya aap {DATEWD} ko kaam karte hain",
            "kya {DOCTOR} {DATEWD} ko clinic mein hote hain",
            "{DOCTOR} kon se din available hote hain",
            "kya aap shaam ko khule hote hain",
            "aap kitne baje tak khule hain",
            "kya {DEPARTMENT} {DATEWD} ko khula hai",
            "kya {DOCTOR} subah ke waqt hote hain",
        ],
    },
    "fees_query": {
        "en": [
            "how much does the {DEPTROLE} charge",
            "can I pay with {PAY}",
            "do you accept {PAY}",
            "what is the fee for {DOCTOR} on {DATEWD}",
        ],
        "ur": [
            "{DEPTROLE} کی فیس کتنی ہے",
            "کیا {PAY} سے ادائیگی ہو سکتی ہے",
            "کیا آپ {PAY} قبول کرتے ہیں",
            "{DATEWD} کو {DOCTOR} کی فیس کتنی ہے",
        ],
        "rur": [
            "{DEPTROLE} ki fees kitni hai",
            "kya {PAY} se payment ho sakti hai",
            "kya aap {PAY} qabool karte hain",
            "{DATEWD} ko {DOCTOR} ki fee kitni hai",
        ],
    },
    "symptom_inquiry": {
        "en": [
            "I have {LOCSYM} in my {BODY}",
            "there is {LOCSYM} in my {BODY} since two days",
            "my {FAMILY} has {SYMPTOM} what should we do",
            "my {FAMILY} has been having {SYMPTOM} and {SYMPTOM}",
        ],
        "ur": [
            "میرے {BODY} میں {LOCSYM} ہے",
            "دو دن سے {BODY} میں {LOCSYM} ہے",
            "{FAMILY} کو {SYMPTOM} ہے ہم کیا کریں",
            "{FAMILY} کو {SYMPTOM} اور {SYMPTOM} ہے",
        ],
        "rur": [
            "mere {BODY} mein {LOCSYM} hai",
            "do din se {BODY} mein {LOCSYM} hai",
            "{FAMILY} ko {SYMPTOM} hai hum kya karein",
            "{FAMILY} ko {SYMPTOM} aur {SYMPTOM} hai",
        ],
    },
    "out_of_scope": {
        "en": [
            "what is the {PRICEITEM} rate",
            "how much is {PRICEITEM} nowadays",
            "what is the price of {PRICEITEM}",
        ],
        "ur": [
            "{PRICEITEM} کا ریٹ کیا ہے",
            "{PRICEITEM} کی قیمت کیا چل رہی ہے",
        ],
        "rur": [
            "{PRICEITEM} ka rate kya hai",
            "{PRICEITEM} ki qeemat kya chal rahi hai",
        ],
    },
}
for _intent, _by_lang in EXTRA_TEMPLATES.items():
    for _lang, _temps in _by_lang.items():
        TEMPLATES[_intent][_lang].extend(_temps)


# --------------------------------------------------------------------------
# v3 ADDITIONS  (from synthetic-test error analysis)
# Root causes found: (a) the template-level split could put EVERY template of a
# pattern (payment methods, day-parts, "what time", "which department") into
# val/test, leaving training with no examples; (b) too few paraphrases.
# Fixes: coverage-aware split (see assign_splits), many more paraphrases,
# hard negatives spread over several intents, past-time / day-part / staff /
# nearby-place slots tagged O, spelling noise.
# --------------------------------------------------------------------------
SLOT_ENTITY.update({"DAYPART": None, "PAST": None, "NEARBY": None, "STAFF": None, "OOSHELP": None})

SLOTS.update({
    "DAYPART": {"en": ["morning", "evening", "afternoon"],
                "ur": ["صبح", "شام", "دوپہر"],
                "rur": ["subah", "shaam", "dopahar"]},
    "PAST": {"en": ["yesterday", "last night", "this morning", "last week", "two days"],
             "ur": ["کل رات", "کل", "صبح", "پچھلے ہفتے", "دو دن"],
             "rur": ["kal raat", "kal", "subah", "pichle hafte", "do din"]},
    "NEARBY": {"en": ["pharmacy", "atm", "mosque", "lab", "taxi stand"],
               "ur": ["فارمیسی", "اے ٹی ایم", "مسجد", "لیب", "ٹیکسی اسٹینڈ"],
               "rur": ["pharmacy", "atm", "masjid", "lab", "taxi stand"]},
    "STAFF": {"en": ["receptionist", "manager", "nurse", "supervisor", "help desk"],
              "ur": ["ریسپشنسٹ", "منیجر", "نرس", "سپروائزر", "ہیلپ ڈیسک"],
              "rur": ["receptionist", "manager", "nurse", "supervisor", "help desk"]},
    "OOSHELP": {"en": ["my exam preparation", "an assignment", "writing a cv", "planning a trip", "cooking dinner"],
                "ur": ["امتحان کی تیاری", "اسائنمنٹ", "سی وی بنانے", "سفر کی منصوبہ بندی", "کھانا پکانے"],
                "rur": ["exam ki tayyari", "assignment", "cv banane", "safar ki planning", "khana pakane"]},
})

EXTRA_V3 = {
    "book_appointment": {
        "en": ["I need an appointment {DATE} in the {DAYPART}",
               "any slot {DATEWD} {DAYPART} with {DOCTOR}",
               "can I come in on {DATEWD} in the {DAYPART}",
               "please arrange a visit with {DEPARTMENT} for my {FAMILY}"],
        "ur": ["مجھے {DATE} {DAYPART} کو اپوائنٹمنٹ چاہیے",
               "{DATEWD} کو {DAYPART} میں {DOCTOR} کا کوئی سلاٹ ہے",
               "کیا میں {DATEWD} کو {DAYPART} میں آ سکتا ہوں",
               "{FAMILY} کے لیے {DEPARTMENT} میں وزٹ کا بندوبست کر دیں"],
        "rur": ["mujhe {DATE} {DAYPART} ko appointment chahiye",
                "{DATEWD} ko {DAYPART} mein {DOCTOR} ka koi slot hai",
                "kya main {DATEWD} ko {DAYPART} mein aa sakta hoon",
                "{FAMILY} ke liye {DEPARTMENT} mein visit ka bandobast kar dein"],
    },
    "cancel_appointment": {
        "en": ["I dont need the appointment anymore",
               "forget about my booking I changed my mind",
               "I wont be coming please remove my slot",
               "please drop my appointment",
               "scrap my {DATE} booking",
               "I want to withdraw my booking with {DOCTOR}"],
        "ur": ["مجھے اپوائنٹمنٹ اب نہیں چاہیے",
               "میری بکنگ رہنے دیں میں نے ارادہ بدل لیا ہے",
               "میں نہیں آ رہا میرا سلاٹ ہٹا دیں",
               "میری اپوائنٹمنٹ ڈراپ کر دیں",
               "{DATE} کی بکنگ ختم کر دیں",
               "{DOCTOR} کے ساتھ اپنی بکنگ واپس لینی ہے"],
        "rur": ["mujhe appointment ab nahi chahiye",
                "meri booking rehne dein maine irada badal liya hai",
                "main nahi aa raha mera slot hata dein",
                "meri appointment drop kar dein",
                "{DATE} ki booking khatam kar dein",
                "{DOCTOR} ke saath apni booking wapas leni hai"],
    },
    "reschedule_appointment": {
        "en": ["can I come at {TIME} instead of {TIME}",
               "please move my slot to a later time",
               "can we do it a day later",
               "is it possible to come earlier than my booking",
               "I want to change the date of my appointment",
               "I need to change my appointment with {DOCTOR} to {DATE}"],
        "ur": ["کیا میں {TIME} کے بجائے {TIME} آ سکتا ہوں",
               "میرا سلاٹ تھوڑا بعد میں کر دیں",
               "کیا اپوائنٹمنٹ ایک دن بعد ہو سکتی ہے",
               "کیا میں بکنگ سے پہلے آ سکتا ہوں",
               "مجھے اپنی اپوائنٹمنٹ کی تاریخ بدلنی ہے",
               "{DOCTOR} کے ساتھ میری اپوائنٹمنٹ {DATE} کر دیں"],
        "rur": ["kya main {TIME} ki jagah {TIME} par aa sakta hoon",
                "mera slot thora baad mein kar dein",
                "kya appointment ek din baad ho sakti hai",
                "kya main booking se pehle aa sakta hoon",
                "mujhe apni appointment ki date badalni hai",
                "{DOCTOR} ke saath meri appointment {DATE} kar dein"],
    },
    "clinic_info": {
        "en": ["is there any {NEARBY} near the clinic",
               "where can i find the nearest {NEARBY}",
               "at what time does the clinic close",
               "what time does {DOCTOR} start seeing patients"],
        "ur": ["کیا کلینک کے قریب کوئی {NEARBY} ہے",
               "آپ کے قریب {NEARBY} کہاں ملے گا",
               "کلینک کتنے بجے بند ہوتا ہے",
               "{DOCTOR} مریضوں کو کتنے بجے دیکھنا شروع کرتے ہیں"],
        "rur": ["kya clinic ke qareeb koi {NEARBY} hai",
                "aap ke qareeb {NEARBY} kahan milega",
                "clinic kitne baje band hota hai",
                "{DOCTOR} marizon ko kitne baje dekhna shuru karte hain"],
    },
    "fees_query": {
        "en": ["is {PAY} accepted for the consultation fee",
               "what is the fee if I pay by {PAY}",
               "how much for a follow up with {DOCTOR}",
               "do you charge extra for the {DEPTROLE}"],
        "ur": ["کیا کنسلٹیشن فیس {PAY} سے لی جاتی ہے",
               "{PAY} سے ادائیگی پر فیس کتنی ہوگی",
               "{DOCTOR} کے ساتھ فالو اپ کی فیس کتنی ہے",
               "کیا {DEPTROLE} کی فیس زیادہ ہے"],
        "rur": ["kya consultation fee {PAY} se li jati hai",
                "{PAY} se payment par fee kitni hogi",
                "{DOCTOR} ke saath follow up ki fee kitni hai",
                "kya {DEPTROLE} ki fees zyada hai"],
    },
    "symptom_inquiry": {
        "en": ["{SYMPTOM} since {PAST} who should I see",
               "which doctor treats {SYMPTOM}",
               "I have been having {SYMPTOM} since {PAST}",
               "my {FAMILY} has had {SYMPTOM} since {PAST}",
               "should I be worried about {SYMPTOM}",
               "{SYMPTOM} is getting worse what should I do"],
        "ur": ["{PAST} سے {SYMPTOM} ہے کس کو دکھاؤں",
               "{SYMPTOM} کا علاج کون سا ڈاکٹر کرتا ہے",
               "مجھے {PAST} سے {SYMPTOM} کی شکایت ہے",
               "{FAMILY} کو {PAST} سے {SYMPTOM} ہے",
               "کیا مجھے {SYMPTOM} کی فکر کرنی چاہیے",
               "{SYMPTOM} میں اضافہ ہو رہا ہے کیا کروں"],
        "rur": ["{PAST} se {SYMPTOM} hai kisko dikhaun",
                "{SYMPTOM} ka ilaaj kaun sa doctor karta hai",
                "mujhe {PAST} se {SYMPTOM} ki shikayat hai",
                "{FAMILY} ko {PAST} se {SYMPTOM} hai",
                "kya mujhe {SYMPTOM} ki fikar karni chahiye",
                "{SYMPTOM} mein izafa ho raha hai kya karun"],
    },
    "talk_to_human": {
        "en": ["can I speak to the {STAFF}", "put me through to the {STAFF}",
               "I want to complain to the {STAFF}", "is the {STAFF} available to talk"],
        "ur": ["مجھے {STAFF} سے بات کرنی ہے", "مجھے {STAFF} سے ملائیں",
               "میں {STAFF} سے شکایت کرنا چاہتا ہوں", "کیا {STAFF} بات کرنے کے لیے دستیاب ہیں"],
        "rur": ["mujhe {STAFF} se baat karni hai", "mujhe {STAFF} se milwaein",
                "main {STAFF} se shikayat karna chahta hoon", "kya {STAFF} baat karne ke liye available hain"],
    },
    "out_of_scope": {
        "en": ["what is the latest news", "suggest a good movie to watch", "how do I learn english quickly",
               "tell me a short story", "book me a flight to lahore", "who won the football match",
               "how tall is mount everest", "what is the stock market doing",
               "translate this sentence into french", "can you help me with {OOSHELP}",
               "order a pizza for me", "what time does the cricket match start",
               "how far is islamabad from karachi", "what is the best phone to buy"],
        "ur": ["تازہ ترین خبریں کیا ہیں", "کوئی اچھی فلم بتائیں", "جلدی انگریزی کیسے سیکھیں",
               "مجھے ایک چھوٹی کہانی سنائیں", "میرے لیے لاہور کی فلائٹ بک کر دیں", "فٹبال کا میچ کون جیتا",
               "ماؤنٹ ایورسٹ کتنا اونچا ہے", "اسٹاک مارکیٹ کا کیا حال ہے",
               "اس جملے کا فرانسیسی میں ترجمہ کریں", "کیا آپ {OOSHELP} میں مدد کر سکتے ہیں",
               "میرے لیے پیزا آرڈر کر دیں", "کرکٹ میچ کتنے بجے شروع ہوتا ہے",
               "اسلام آباد کراچی سے کتنا دور ہے", "سب سے اچھا موبائل کون سا ہے"],
        "rur": ["taaza khabrein kya hain", "koi achi film batao", "jaldi english kaise seekhein",
                "mujhe ek chhoti kahani sunao", "mere liye lahore ki flight book kar do", "football match kaun jeeta",
                "mount everest kitna uncha hai", "stock market ka kya haal hai",
                "is jumle ka french mein tarjuma karo", "kya aap {OOSHELP} mein madad kar sakte hain",
                "mere liye pizza order kar do", "cricket match kitne baje shuru hota hai",
                "islamabad karachi se kitna door hai", "sabse acha mobile kaun sa hai"],
    },
}
for _intent, _by_lang in EXTRA_V3.items():
    for _lang, _temps in _by_lang.items():
        TEMPLATES[_intent][_lang].extend(_temps)

NOISE_PROB = 0.20         # fraction of en/rur utterances that get spelling noise
TOKEN_NOISE_PROB = 0.12   # per-token probability inside a noisy utterance

SLOT_RE = re.compile(r"^\{([A-Z]+)\}$")


def slot_types(template: str):
    return [SLOT_RE.match(t).group(1) for t in template.split() if SLOT_RE.match(t)]



def noisy_token(tok, rng):
    """Cheap typo model for Latin-script words: drop a vowel, double a letter, or swap two."""
    if not (tok.isascii() and tok.isalpha()) or len(tok) < 4:
        return tok
    i = rng.randrange(1, len(tok))
    r = rng.random()
    if r < 0.45 and tok[i] in "aeiou":
        return tok[:i] + tok[i + 1:]
    if r < 0.75:
        return tok[:i] + tok[i] + tok[i:]
    if r < 0.90 and i < len(tok) - 1:
        return tok[:i] + tok[i + 1] + tok[i] + tok[i + 2:]
    return tok


# Patterns that must stay represented in TRAIN in every language (found via error analysis).
TRIGGERS = {
    "daypart": r"(morning|evening|afternoon|subah|shaam|dopahar|صبح|شام|دوپہر)",
    "time_q": r"(what time|at what time|kitne baje|کتنے بجے)",
    "department_q": r"(department|شعبے)",
    "help_q": r"(help|madad|مدد)",
}
MIN_TRAIN_PER_FEATURE = 2


def template_features(t):
    feats = set(slot_types(t))
    low = re.sub(r"\{[A-Z]+\}", " ", t).lower()   # match literal words only, not slot names
    for name, rx in TRIGGERS.items():
        if re.search(rx, low):
            feats.add("trig:" + name)
    return feats


def assign_splits(items, rng):
    """Coverage-aware template split for ONE language.
    items = [(intent, template)]. Returns {(intent, template): 'train'|'val'|'test'}.
    A template is moved out of train only if every feature it carries (slot types and
    TRIGGERS) keeps >= MIN_TRAIN_PER_FEATURE templates in train."""
    feats = {it: template_features(it[1]) for it in items}
    train_count = {}
    for it in items:
        for f in feats[it]:
            train_count[f] = train_count.get(f, 0) + 1
    by_intent = {}
    for it in items:
        by_intent.setdefault(it[0], []).append(it)
    quota = {}
    for intent, lst in by_intent.items():
        n = len(lst)
        quota[intent] = {"test": max(2, round(0.2 * n)), "val": max(1, round(0.1 * n))}
    got = {intent: {"test": 0, "val": 0} for intent in by_intent}
    split = {it: "train" for it in items}
    order = list(items)
    rng.shuffle(order)
    for it in order:
        intent = it[0]
        for sp in ("test", "val"):
            if got[intent][sp] < quota[intent][sp]:
                if all(train_count[f] - 1 >= MIN_TRAIN_PER_FEATURE for f in feats[it]):
                    split[it] = sp
                    got[intent][sp] += 1
                    for f in feats[it]:
                        train_count[f] -= 1
                break
    return split


def validate_templates():
    """Fail fast on malformed templates (braces glued to punctuation, unknown slots)."""
    for intent, by_lang in TEMPLATES.items():
        assert set(by_lang) == set(LANGS), f"{intent}: missing language"
        for lang, temps in by_lang.items():
            for t in temps:
                for tok in t.split():
                    if "{" in tok or "}" in tok:
                        m = SLOT_RE.match(tok)
                        assert m, f"Malformed slot token {tok!r} in {intent}/{lang}: {t}"
                        assert m.group(1) in SLOTS, f"Unknown slot {tok} in {t}"
                types = slot_types(t)
                for ty in set(types):
                    assert types.count(ty) <= len(SLOTS[ty][lang]), t


def fill_all(template: str, lang: str):
    """All distinct fillings (values for repeated slot types are sampled w/o replacement)."""
    types = slot_types(template)
    if not types:
        return [[]]
    pools = [SLOTS[ty][lang] for ty in types]
    combos = []
    for combo in product(*pools):
        ok = True
        for ty in set(types):
            vals = [combo[i] for i, t in enumerate(types) if t == ty]
            if len(vals) != len(set(vals)):
                ok = False
                break
        if ok:
            combos.append(list(combo))
    return combos


def build_example(template: str, values, prefix=None, suffix=None):
    tokens, tags = [], []
    if prefix:
        for w in prefix.split():
            tokens.append(w)
            tags.append("O")
    vi = 0
    types = slot_types(template)
    for tok in template.split():
        m = SLOT_RE.match(tok)
        if m:
            ty = types[vi]
            ent = SLOT_ENTITY.get(ty, ty)
            words = values[vi].split()
            vi += 1
            for k, w in enumerate(words):
                tokens.append(w)
                tags.append("O" if ent is None else ("B-" if k == 0 else "I-") + ent)
        else:
            tokens.append(tok)
            tags.append("O")
    if suffix:
        for w in suffix.split():
            tokens.append(w)
            tags.append("O")
    return tokens, tags


def split_templates(templates, rng):
    n = len(templates)
    idx = list(range(n))
    rng.shuffle(idx)
    n_test = max(1, round(n * 0.2))
    n_val = max(1, round(n * 0.1))
    test = [templates[i] for i in idx[:n_test]]
    val = [templates[i] for i in idx[n_test:n_test + n_val]]
    train = [templates[i] for i in idx[n_test + n_val:]]
    return {"train": train, "val": val, "test": test}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-target", type=int, default=90, help="examples per (intent, language)")
    ap.add_argument("--val-target", type=int, default=12)
    ap.add_argument("--test-target", type=int, default=20)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    validate_templates()
    rng = random.Random(args.seed)
    out = Path(args.out)
    (out / "synthetic").mkdir(parents=True, exist_ok=True)

    target = {"train": args.train_target, "val": args.val_target, "test": args.test_target}
    records = {"train": [], "val": [], "test": []}
    seen = {"train": set(), "val": set(), "test": set()}

    split_of = {}
    for lang in LANGS:
        items = [(i, t) for i, bl in TEMPLATES.items() for t in bl[lang]]
        split_of[lang] = assign_splits(items, rng)
    used_templates = {sp: {l: set() for l in LANGS} for sp in ("train", "val", "test")}

    for intent, by_lang in TEMPLATES.items():
        for lang in LANGS:
            parts = {"train": [], "val": [], "test": []}
            for t in by_lang[lang]:
                parts[split_of[lang][(intent, t)]].append(t)
            for split, temps in parts.items():
                if not temps:
                    print(f"[warn] no {split} templates for {intent}/{lang}")
                    continue
                quota = target[split]
                per_t = -(-quota // len(temps))
                bucket = []
                for t in temps:
                    combos = fill_all(t, lang)
                    made, attempts = 0, 0
                    while made < per_t and attempts < per_t * 40:
                        attempts += 1
                        values = rng.choice(combos)
                        prefix = rng.choice(PREFIXES[lang]) if rng.random() < PREFIX_PROB else None
                        suffix = rng.choice(SUFFIXES[lang]) if rng.random() < SUFFIX_PROB else None
                        tokens, tags = build_example(t, values, prefix, suffix)
                        if rng.random() < LOWER_PROB:
                            tokens = [w.lower() for w in tokens]
                        if lang in ("en", "rur") and rng.random() < NOISE_PROB:
                            tokens = [noisy_token(w, rng) if rng.random() < TOKEN_NOISE_PROB else w
                                      for w in tokens]
                        text = " ".join(tokens)
                        if text in seen[split]:
                            continue
                        seen[split].add(text)
                        made += 1
                        used_templates[split][lang].add((intent, t))
                        bucket.append({"lang": lang, "intent": intent, "text": text,
                                       "tokens": tokens, "ner_tags": tags})
                rng.shuffle(bucket)
                records[split].extend(bucket[:quota])

    # Guard against any text appearing in more than one split (leakage)
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = seen[a] & seen[b]
        if overlap:
            # drop from the *evaluation* split so training stays intact
            drop_from = b
            records[drop_from] = [r for r in records[drop_from] if r["text"] not in overlap]
            print(f"[warn] removed {len(overlap)} overlapping texts from {drop_from}")

    stats = {}
    for split, recs in records.items():
        rng.shuffle(recs)
        for i, r in enumerate(recs):
            r["id"] = f"{split}-{i:05d}"
        with open(out / "synthetic" / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[split] = {
            "n": len(recs),
            "by_lang": dict(Counter(r["lang"] for r in recs)),
            "by_intent": dict(Counter(r["intent"] for r in recs)),
            "distinct_templates": {l: len(used_templates[split][l]) for l in LANGS},
        }

    intents = list(TEMPLATES.keys())
    ner_labels = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")]
    with open(out / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({
            "intents": intents,
            "intent2id": {k: i for i, k in enumerate(intents)},
            "ner_labels": ner_labels,
            "ner2id": {k: i for i, k in enumerate(ner_labels)},
        }, f, ensure_ascii=False, indent=2)
    with open(out / "synthetic" / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    for k, v in stats.items():
        print(f"{k:<6} n={v['n']:<5} by_lang={v['by_lang']}  distinct_templates={v['distinct_templates']}")


if __name__ == "__main__":
    main()
