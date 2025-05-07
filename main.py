
import re
import torch
import unicodedata
import difflib  # Added import
from difflib import SequenceMatcher  # Explicitly import SequenceMatcher
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, MBart50TokenizerFast, MBartForConditionalGeneration

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and tokenizers
# MT5 for paragraphs (renamed endpoint)
mt5_paragraph_model_path = "RanawahajAhmed/finetuned_mt5_large_for_urdu_correction"
mt5_paragraph_tokenizer = MT5Tokenizer.from_pretrained(mt5_paragraph_model_path, use_fast=False)
mt5_paragraph_model = MT5ForConditionalGeneration.from_pretrained(mt5_paragraph_model_path)

# MT5 for sentences
mt5_sentence_model_path = "RanawahajAhmed/mt5_finetuned_for_urdu_sentence_correction"
mt5_sentence_model = MT5ForConditionalGeneration.from_pretrained(mt5_sentence_model_path)
mt5_sentence_tokenizer = mt5_paragraph_tokenizer  # Reuse the existing MT5 tokenizer

# mBART for paragraphs
bart_paragraph_model_path = "RanawahajAhmed/mbart_finetuned_for_urdu_paragraphs_correction"
print(f"Loading tokenizer from {bart_paragraph_model_path}...")
bart_paragraph_tokenizer = MBart50TokenizerFast.from_pretrained(bart_paragraph_model_path)
urdu_lang_id = bart_paragraph_tokenizer.lang_code_to_id.get("ur_PK", 250054)  # Default to 250054 if ur_PK not found
print("Tokenizer loaded successfully.")
print(f"Loading model from {bart_paragraph_model_path}...")
bart_paragraph_model = MBartForConditionalGeneration.from_pretrained(bart_paragraph_model_path)
print("Model loaded successfully.")

# Error dictionaries (same as provided)
noun_errors = {
    "پاکستان": "پاکیستان", "لوگ": "لوک", "مسائل": "مسایل", "مہنگائی": "مہنگای", "حکومت": "حوکمت",
    "تعلیم": "تعلم", "صحت": "سحت", "گاؤں": "گاؤن", "موسم": "موسوم", "ٹیکنالوجی": "ٹکنالوجی",
    "پانی": "پانے", "کرپشن": "کرفشن", "روزگار": "روزگاز", "سیاحت": "سایاحت", "نوجوان": "نووجوان",
    "زراعت": "زرات", "صنعت": "سنعت", "سیلاب": "سیلاپ", "ٹرانسپورٹ": "ٹرانسپوٹ", "خواتین": "خواتیں",
    "کھیل": "کحیل", "آلودگی": "الودگی", "امید": "امد", "دوستی": "دوسٹی", "زندگی": "زندکی"
}

verb_errors = {
    "کرتے": "کرو", "بتا": "بولو", "بڑھ": "چڑھ", "ہو": "جا", "آتی": "جاتی", "رکھنا": "اٹھانا",
    "سوچنا": "سونگھنا", "ملتی": "ٹلتی", "آزماتے": "چکھتے", "نکلتی": "پھسلتی", "دیتا": "بیچتا",
    "بدل": "پھٹ", "رکھتی": "پکڑتی", "آتا": "بھاگتا", "چھوڑ": "پھوڑ", "لینا": "دھونا", "دیکھنا": "چومنا",
    "سمجھنا": "چلنا", "پڑھنا": "گھومنا", "چلنا": "ہنسنا", "لگایا": "جلایا", "کمایا": "سمجھایا",
    "کیا": "پیا", "دیا": "کیا", "کیں": "سیں", "آئیں": "جائیں", "دیں": "لیں", "کریں": "سریں"
}

pronoun_errors = {
    "وہ": "یہ", "ہم": "تم", "اس": "ان", "ان": "اس", "ہمارے": "تمہارے", "یہ": "وہ", "کسی": "کوئی",
    "جو": "جس", "ہر": "کوئی", "اپنی": "ان کی", "تم": "ہم", "میں": "ہم", "ہمیں": "تمہیں", "انہوں": "اس نے",
    "کس": "کیا", "کچھ": "سب", "کوئی": "ہر", "خود": "دوسرے", "ان کا": "ہمارا", "اپنوں": "دوسروں",
    "ہم سب": "تم سب", "ان سب": "اس سب", "کس نے": "کیا نے", "جس نے": "جو نے", "جو کچھ": "کچھ بھی",
    "ہر ایک": "کوئی ایک", "ان کے": "ہمارے", "ہم نے": "تم نے", "تم نے": "ہم نے"
}

prepositions_conjunctions_errors = {
    "میں": "سے", "سے": "میں", "کے": "پر", "پر": "کے", "اور": "یا", "تا کہ": "کیونکہ", "کی": "کا",
    "کہ": "جو", "لیے": "بغیر", "بھی": "نہ", "جس": "کہ", "اگر": "ورنہ", "تو": "مگر", "جب": "کہاں",
    "تک": "سے", "چونکہ": "تاکہ", "کیونکہ": "اگر", "جو": "کہ", "جن": "جس", "ورنہ": "اگر", "یا": "اور",
    "مگر": "تو", "بل کہ": "بلکہ", "جب کہ": "کیونکہ"
}

object_errors = {
    "مسائل": "مسایئل", "مہنگائی": "مہنگای", "تعلیم": "تلیم", "صحت": "سحت", "پانی": "پانے", "کرپشن": "کرپسشن",
    "روزگار": "روزگارد", "سیاحت": "سیاحٹ", "نوجوان": "نواجوان", "زراعت": "زراعٹ", "صنعت": "سنعت",
    "سیلاب": "سیلب", "ٹرانسپورٹ": "ٹرانسپوٹ", "خواتین": "خواتیں", "کھیل": "خیل", "آلودگی": "آلدگی",
    "امید": "امد", "دوستی": "دوستے", "زندگی": "زندگے", "سہولتیں": "سہولتین", "اصلاحات": "اصلحات",
    "سرمایہ": "سرمائہ", "منصوبے": "منصوبہ", "حقوق": "حقووق", "وسائل": "وسائیل"
}

past_tense_verb_errors = {
    "بڑھ گئی": "بڑھتا", "ہو گیا": "ہوتا", "متاثر ہوئی": "متاثر ہوتا", "آ گئے": "آتا", "کہا": "کہتا",
    "بتایا": "بتاتا", "بنائے گئے": "بناتا", "دیا": "دیتا", "پڑھا": "پڑھتا", "چلا": "چلتا", "لگا": "لگتا",
    "ہوئے": "ہوتا", "رکھا": "رکھتا", "کیا": "کرتا", "آئی": "آتا", "دیکھا": "دیکھتا", "سمجھا": "سمجھتا",
    "چھوڑا": "چھوڑتا", "ہوئیں": "ہوتا", "پہنچا": "پہنچتا", "مل گیا": "ملتا", "بدل گیا": "بدلتا",
    "کمایا": "کماتا", "لگایا": "لگاتا", "بنا": "بناتا", "بڑھائی": "بڑھاتا", "پایا": "پاتا", "کیے": "کرتا",
    "سوچا": "سوچتا", "آزمایا": "آزماتا", "نکلا": "نکلتا", "بدلا": "بدلتا", "آیا": "آتا", "لیا": "لیتا",
    "رکھے": "رکھتا", "کیں": "کرتا", "آئیں": "آتا", "دیں": "دیتا", "بڑھائیں": "بڑھاتا", "پائیں": "پاتا",
    "کریں": "کرتا", "ہوئی": "ہوتا", "گئی": "جاتا"
}

present_tense_verb_errors = {
    "کرتے ہیں": "کریں گے", "بتا رہا ہے": "بتائے گا", "بڑھ رہی ہے": "بڑھے گی", "ہوتا ہے": "ہو گا",
    "لگتا ہے": "لگے گا", "چاہیے": "چاہیے گا", "دینی ہے": "دے گا", "متاثر ہو رہا ہے": "متاثر ہو گا",
    "پڑتا ہے": "پڑے گا", "حاصل کر رہے ہیں": "حاصل کریں گے", "دیتی ہے": "دے گی", "بناتے ہیں": "بنائیں گے",
    "آتی ہے": "آئے گی", "رکھتا ہے": "رکھے گا", "سوچتے ہیں": "سوچیں گے", "ملتی ہے": "ملے گی",
    "آزماتے ہیں": "آزمائیں گے", "نکلتی ہے": "نکلے گی", "دیتا ہے": "دے گا", "بدلتے ہیں": "بدلیں گے",
    "رکھتی ہے": "رکھے گی", "آتا ہے": "آئے گا", "چھوڑتے ہیں": "چھوڑیں گے", "لیتے ہیں": "لیں گے",
    "دیکھتے ہیں": "دیکھیں گے", "سمجھتے ہیں": "سمجھیں گے", "پڑھتے ہیں": "پڑھیں گے", "چلتے ہیں": "چلیں گے",
    "لگاتے ہیں": "لگائیں گے", "کماتے ہیں": "کمائیں گے", "رکھتے ہیں": "رکھیں گے", "کرتی ہے": "کرے گی",
    "بتاتی ہے": "بتائے گی", "بڑھتا ہے": "بڑھے گا", "لگتی ہے": "لگے گی", "دیتے ہیں": "دیں گے",
    "بناتی ہے": "بنائے گی", "آتے ہیں": "آئیں گے", "رکھتی ہیں": "رکھیں گی", "سوچتی ہے": "سوچے گی",
    "ملتا ہے": "ملے گا", "آزماتی ہے": "آزمائے گی", "نکلتے ہیں": "نکلیں گے", "بدلتی ہے": "بدلے گی",
    "چھوڑتی ہے": "چھوڑے گی", "لیتی ہے": "لے گی", "دیکھتی ہے": "دیکھے گی", "سمجھتی ہے": "سمجھے گی",
    "پڑھتی ہے": "پڑھے گی", "چلتی ہے": "چلے گی", "لگاتی ہے": "لگائے گی", "کماتی ہے": "کمائے گی",
    "کر رہے ہیں": "کریں گے", "بتا رہے ہیں": "بتائیں گے", "بڑھ رہے ہیں": "بڑھیں گے", "لگ رہے ہیں": "لگیں گے"
}

future_tense_verb_errors = {
    "ہو گا": "ہوا", "کر سکتے ہیں": "کر سکتے تھے", "آئے گا": "آیا", "دیں گے": "دیا", "بنا سکتے ہیں": "بنا سکتے تھے",
    "پائیں گے": "پایا", "ملے گا": "ملا", "ہوں گی": "تھیں", "کریں گے": "کیا", "بڑھے گا": "بڑھا",
    "رکھ سکتے ہیں": "رکھ سکتے تھے", "سوچ سکتے ہیں": "سوچ سکتے تھے", "بدل سکتے ہیں": "بدل سکتے تھے",
    "چھوڑیں گے": "چھوڑا", "لیں گے": "لیا", "آئیں گے": "آئے", "پڑھ سکتے ہیں": "پڑھ سکتے تھے",
    "دیکھیں گے": "دیکھا", "سمجھیں گے": "سمجھا", "ہو سکتے ہیں": "ہو سکتے تھے", "لگے گا": "لگا",
    "کمایا جائے گا": "کمایا گیا", "لگایا جائے گا": "لگایا گیا", "بنایا جائے گا": "بنایا گیا",
    "بڑھایا جائے گا": "بڑھایا گیا", "پایا جائے گا": "پایا گیا", "کیا جائے گا": "کیا گیا",
    "رکھا جائے گا": "رکھا گیا", "دیا جائے گا": "دیا گیا"
}

singular_errors = {
    "پاکستان": "پاکستن", "دوست": "دووست", "مسئلہ": "مسئیلہ", "مہنگائی": "مہنگای", "حکومت": "حکومٹ",
    "تعلیم": "تلیم", "صحت": "سحت", "گاؤں": "گاؤن", "موسم": "موسسم", "ٹیکنالوجی": "ٹکنالوجی",
    "پانی": "پانے", "کرپشن": "کرپسشن", "روزگار": "روزگارد", "سیاحت": "سیاحٹ", "نوجوان": "نواجوان",
    "زراعت": "زراعٹ", "صنعت": "سنعت", "سیلاب": "سیلب", "ٹرانسپورٹ": "ٹرانسپرٹ", "خاتون": "خاتوون",
    "کھیل": "خیل", "آلودگی": "آلدگی", "امید": "امد", "دوستی": "دوستے", "زندگی": "زندگے"
}

plural_errors = {
    "لوگ": "لوک", "مسائل": "مسایئل", "اخراجات": "اخرجت", "سہولتیں": "سہولتین", "ادارے": "ادارہ",
    "طلبہ": "طلباء", "والدین": "والداین", "نوجوانوں": "نواجوانوں", "موسموں": "موسسموں", "مہارتیں": "مہارتین",
    "وسائل": "وسائیل", "مواقع": "مواقق", "اصلاحات": "اصلحات", "منصوبوں": "منصوبہ", "حقوق": "حقووق",
    "کھیلوں": "خیلوں", "شہروں": "شہرن", "بچوں": "بچن", "پروگرامز": "پروگرمز", "کاروباروں": "کارباروں",
    "سیاحوں": "سیاحن", "مریضوں": "مریضن", "پالیسیوں": "پالسیوں", "اتفاقیات": "اتفقیات", "یادیں": "یادین"
}

masculine_errors = {
    "دوست": "دووست", "شخص": "شخس", "نوجوان": "نواجوان", "طالب": "طلیب", "والد": "والدد", "کسان": "کسسن",
    "سیاح": "سیح", "مریض": "مرض", "ڈاکٹر": "ڈکٹر", "سیاستدان": "سیاسدان", "گاؤں": "گاؤن", "شہر": "شہہر",
    "موسم": "موسسم", "روزگار": "روزگارد", "کاروبار": "کاربار", "منصوبہ": "منصوبہہ", "کھیل": "خیل",
    "پانی": "پانے", "ماحول": "محول", "وسائل": "وسائیل", "امن": "امم", "صبر": "سبر", "خواب": "خوب",
    "رشتہ": "رشہ", "چیلنج": "چلنج"
}

feminine_errors = {
    "مہنگائی": "مہنگای", "حکومت": "حکومٹ", "تعلیم": "تلیم", "صحت": "سحت", "ٹیکنالوجی": "ٹکنالوجی",
    "کرپشن": "کرپسشن", "سیاحت": "سیاحٹ", "زراعت": "زراعٹ", "صنعت": "سنعت", "آلودگی": "آلدگی",
    "امید": "امد", "دوستی": "دوستے", "زندگی": "زندگے", "سہولت": "سہوللت", "اصلاح": "اصلح",
    "سرمایہ": "سرمائہ", "پالیسی": "پالسی", "خاتون": "خاتوون", "مہارت": "محارت", "کوشش": "کوشس",
    "مسکراہٹ": "مسکاہٹ", "خاموشی": "خموشی", "خوشی": "خشی", "یاد": "یادد", "ترقی": "ترقے"
}

direct_object_errors = {
    "مسائل": "مسایئل", "مہنگائی": "مہنگای", "تعلیم": "تلیم", "صحت": "سحت", "پانی": "پانے", "کرپشن": "کرپسشن",
    "روزگار": "روزگارد", "سیاحت": "سیاحٹ", "نوجوان": "نواجوان", "زراعت": "زراعٹ", "صنعت": "سنعت",
    "سیلاب": "سیلب", "ٹرانسپورٹ": "ٹرانسپرٹ", "خواتین": "خواتیں", "کھیل": "خیل", "آلودگی": "آلدگی",
    "امید": "امد", "دوستی": "دوستے", "زندگی": "زندگے", "سہولتیں": "سہولتین", "اصلاحات": "اصلحات",
    "سرمایہ": "سرمائہ", "منصوبے": "منصوبہ", "حقوق": "حقووق", "وسائل": "وسائیل"
}

indirect_object_errors = {
    "پاکستان میں": "پاکستن میں", "گاؤں میں": "گاؤن میں", "شہر میں": "شہہر میں", "مارکیٹ میں": "مارکیٹٹ میں",
    "اسکولوں میں": "اسکولن میں", "اسپتالوں میں": "اسپتلن میں", "شعبوں میں": "شعبن میں", "میدانوں میں": "میدانن میں",
    "اداروں میں": "ادارن میں", "منصوبوں میں": "منصوبن میں", "موسموں میں": "موسسموں میں", "حالات میں": "حالت میں",
    "زندگی میں": "زندگے میں", "معاشرے میں": "معاشرہ میں", "دور میں": "دورر میں", "وقت میں": "وققت میں",
    "ماحول میں": "محول میں", "سہولتوں میں": "سہولتین میں", "مسائل میں": "مسایئل میں", "حقوق میں": "حقووق میں",
    "ترقی میں": "ترقے میں", "امن میں": "امم میں", "خوابوں میں": "خوابن میں", "امید میں": "امد میں",
    "دوستی میں": "دوستے میں"
}

adverb_errors = {
    "زیادہ": "زیادہہ", "فوری": "فورری", "سست": "سسٹ", "برابر": "برابار", "مناسب طور پر": "مناسب طورر پر",
    "شدید": "شدیدد", "آہستہ": "آہستہہ", "ہمیشہ": "ہمشہ", "کبھی": "کبھے", "ابھی": "ابھے", "پہلے": "پہلہ",
    "بعد": "بدد", "صرف": "سررف", "بہت": "بہہت", "کم": "ککم", "دوبارہ": "دوبارہہ", "اکثر": "اکسر",
    "ہر سال": "ہرر سال", "جلدی": "جلدے", "صحیح": "سحیح", "واضح": "واضضح", "خاموشی سے": "خموشی سے",
    "درست": "درسٹ", "سادہ": "سادہہ"
}

possessive_noun_errors = {
    "پاکستان کی": "پاکستن کی", "لوگوں کی": "لوکوں کی", "حکومت کی": "حکومٹ کی", "تعلیم کی": "تلیم کی",
    "صحت کی": "سحت کی", "گاؤں کی": "گاؤن کی", "موسم کی": "موسسم کی", "ٹیکنالوجی کی": "ٹکنالوجی کی",
    "پانی کی": "پانے کی", "کرپشن کی": "کرپسشن کی", "روزگار کی": "روزگارد کی", "سیاحت کی": "سیاحٹ کی",
    "نوجوانوں کی": "نواجوانوں کی", "زراعت کی": "زراعٹ کی", "صنعت کی": "سنعت کی", "سیلاب کی": "سیلب کی",
    "ٹرانسپورٹ کی": "ٹرانسپرٹ کی", "خواتین کی": "خواتیں کی", "کھیل کی": "خیل کی", "آلودگی کی": "آلدگی کی",
    "امید کی": "امد کی", "دوستی کی": "دوستے کی", "زندگی کی": "زندگے کی", "سہولت کی": "سہوللت کی",
    "اصلاح کی": "اصلح کی"
}

possessed_noun_errors = {
    "معیشت": "معشیت", "ترقی": "ترقے", "زندگی": "زندگے", "سہولتیں": "سہولتین", "حالت": "حالتت",
    "پیداوار": "پیدوار", "سرمایہ": "سرمائہ", "پالیسی": "پالسی", "مہارت": "محارت", "کوشش": "کوشس",
    "مسکراہٹ": "مسکاہٹ", "خاموشی": "خموشی", "خوشی": "خشی", "یاد": "یادد", "امید": "امد", "دوستی": "دوستے",
    "سادگی": "سادگے", "سکون": "سکوون", "احساس": "احسس", "شخصیت": "شخسیت", "بنیاد": "بنیادد",
    "رفتاری": "رفتارے", "چیزوں": "چیزن", "اعتماد": "اعتمد"
}

sentence_component_errors = {
    "پاکستان": "پاکستن", "لوگ": "لوک", "کرتے": "کریت", "مسائل": "مسایئل", "مہنگائی": "مہنگای",
    "حکومت": "حکومٹ", "تعلیم": "تلیم", "صحت": "سحت", "گاؤں": "گاؤن", "موسم": "موسسم", "ٹیکنالوجی": "ٹکنالوجی",
    "پانی": "پانے", "کرپشن": "کرپسشن", "روزگار": "روزگارد", "سیاحت": "سیاحٹ", "نوجوان": "نواجوان",
    "زراعت": "زراعٹ", "صنعت": "سنعت", "سیلاب": "سیلب", "ٹرانسپورٹ": "ٹرانسپرٹ", "خواتین": "خواتیں",
    "کھیل": "خیل", "آلودگی": "آلدگی", "امید": "امد", "دوستی": "دوستے"
}

subject_errors = {
    "لوگ": "لوک", "دوست": "دووست", "شخص": "شخس", "حکومت": "حکومٹ", "نوجوان": "نواجوان", "والدین": "والداین",
    "کسان": "کسسن", "سیاح": "سیح", "مریض": "مرض", "ڈاکٹر": "ڈکٹر", "سیاستدان": "سیاسدان", "طلبہ": "طلباء",
    "خواتین": "خواتیں", "پاکستان": "پاکستن", "گاؤں": "گاؤن", "شہر": "شہہر", "موسم": "موسسم",
    "ٹیکنالوجی": "ٹکنالوجی", "پانی": "پانے", "کرپشن": "کرپسشن", "روزگار": "روزگارد", "سیاحت": "سیاحٹ",
    "زراعت": "زراعٹ", "صنعت": "سنعت", "آلودگی": "آلدگی"
}

verb_object_errors = {
    "مسائل": "مسایئل", "مہنگائی": "مہنگای", "تعلیم": "تلیم", "صحت": "سحت", "پانی": "پانے", "کرپشن": "کرپسشن",
    "روزگار": "روزگارد", "سیاحت": "سیاحٹ", "نوجوان": "نواجوان", "زراعت": "زراعٹ", "صنعت": "سنعت",
    "سیلاب": "سیلب", "ٹرانسپورٹ": "ٹرانسپرٹ", "خواتین": "خواتیں", "کھیل": "خیل", "آلودگی": "آلدگی",
    "امید": "امد", "دوستی": "دوستے", "زندگی": "زندگے", "سہولتیں": "سہولتین", "اصلاحات": "اصلحات",
    "سرمایہ": "سرمائہ", "منصوبے": "منصوبہ", "حقوق": "حقووق", "وسائل": "وسائیل"
}

# Combine all dictionaries into a list for sequential processing
error_dictionaries = [
    ("noun_errors", noun_errors, "اسم کی غلطی", "یہ اسم کی غلطی ہے کیونکہ اس کی ہجے غلط ہے یا اسے غلط استعمال کیا گیا ہے۔"),
    ("verb_errors", verb_errors, "فعل کی غلطی", "یہ فعل کی غلطی ہے کیونکہ یہ غلط زمانے یا شکل میں ہے۔"),
    ("pronoun_errors", pronoun_errors, "ضمیر کی غلطی", "یہ ضمیر کی غلطی ہے کیونکہ یہ غلط فاعل یا مفعول کی طرف اشارہ کرتا ہے۔"),
    ("prepositions_conjunctions_errors", prepositions_conjunctions_errors, "حرف ربط یا حرف جار کی غلطی", "یہ حرف ربط یا حرف جار کی غلطی ہے کیونکہ اسے جملے میں غلط استعمال کیا گیا ہے۔"),
    ("object_errors", object_errors, "مفعول کی غلطی", "یہ مفعول کی غلطی ہے کیونکہ مفعول کی ہجے غلط ہے یا غلط ہے۔"),
    ("past_tense_verb_errors", past_tense_verb_errors, "فعل ماضی کی غلطی", "یہ فعل ماضی کی غلطی ہے کیونکہ یہ غلط شکل میں ہے۔"),
    ("present_tense_verb_errors", present_tense_verb_errors, "فعل حال کی غلطی", "یہ فعل حال کی غلطی ہے کیونکہ یہ غلط شکل میں ہے۔"),
    ("future_tense_verb_errors", future_tense_verb_errors, "فعل مستقبل کی غلطی", "یہ فعل مستقبل کی غلطی ہے کیونکہ یہ غلط شکل میں ہے۔"),
    ("singular_errors", singular_errors, "واحد کی غلطی", "یہ واحد اسم کی غلطی ہے کیونکہ اس کی ہجے غلط ہے یا اسے غلط استعمال کیا گیا ہے۔"),
    ("plural_errors", plural_errors, "جمع کی غلطی", "یہ جمع اسم کی غلطی ہے کیونکہ اس کی ہجے غلط ہے یا اسے غلط استعمال کیا گیا ہے۔"),
    ("masculine_errors", masculine_errors, "مذکر کی غلطی", "یہ مذکر اسم کی غلطی ہے کیونکہ اس کی ہجے غلط ہے یا اسے غلط استعمال کیا گیا ہے۔"),
    ("feminine_errors", feminine_errors, "مؤnث کی غلطی", "یہ مؤnث اسم کی غلطی ہے کیونکہ اس کی ہجے غلط ہے یا اسے غلط استعمال کیا گیا ہے۔"),
    ("direct_object_errors", direct_object_errors, "مفعول مستقیم کی غلطی", "یہ مفعول مستقیم کی غلطی ہے کیونکہ اس کی ہجے غلط ہے یا غلط ہے۔"),
    ("indirect_object_errors", indirect_object_errors, "مفعول غیر مستقیم کی غلطی", "یہ مفعول غیر مستقیم کی غلطی ہے کیونکہ اسے غلط استعمال کیا گیا ہے۔"),
    ("adverb_errors", adverb_errors, "متعلق فعل کی غلطی", "یہ متعلق فعل کی غلطی ہے کیونکہ اس کی ہجے غلط ہے یا اسے غلط استعمال کیا گیا ہے۔"),
    ("possessive_noun_errors", possessive_noun_errors, "اسم ملکیتی کی غلطی", "یہ اسم ملکیتی کی غلطی ہے کیونکہ اس کی ہجے غلط ہے یا غلط ہے۔"),
    ("possessed_noun_errors", possessed_noun_errors, "اسم مملوک کی غلطی", "یہ اسم مملوک کی غلطی ہے کیونکہ اس کی ہجے غلط ہے یا غلط ہے۔"),
    ("sentence_component_errors", sentence_component_errors, "جملہ کے جزو کی غلطی", "یہ جملہ کے جزو کی غلطی ہے کیونکہ اسے غلط استعمال کیا گیا ہے۔"),
    ("subject_errors", subject_errors, "فاعل کی غلطی", "یہ فاعل کی غلطی ہے کیونکہ اس کی ہجے غلط ہے یا غلط ہے۔"),
    ("verb_object_errors", verb_object_errors, "فعل اور مفعول کی غلطی", "یہ فعل اور مفعول کی غلطی ہے کیونکہ مفعول فعل کے ساتھ مطابقت نہیں رکھتا۔")
]

def detect_errors(input_text, corrected_text):
    """
    Detect errors by comparing input text with corrected text using sequence alignment.
    Identifies errors corrected by the model according to error dictionaries.
    Returns a list of unique detected errors with explanations and reasons in Urdu.
    """
    detected_errors = []
    seen_errors = set()  # To avoid duplicates

    # Normalize Unicode to handle encoding variations
    input_text = unicodedata.normalize('NFC', input_text)
    corrected_text = unicodedata.normalize('NFC', corrected_text)

    # Tokenize input and corrected text into words
    input_words = re.findall(r'\S+', input_text)
    corrected_words = re.findall(r'\S+', corrected_text)

    # Log tokenized words for debugging
    print("Input words:", input_words)
    print("Corrected words:", corrected_words)

    # Use SequenceMatcher to align input and corrected words
    matcher = SequenceMatcher(None, input_words, corrected_words)
    matches = matcher.get_opcodes()

    # Log alignment operations
    print("Alignment operations:", matches)

    # Process alignment operations
    for tag, i1, i2, j1, j2 in matches:
        if tag == 'equal':
            # Words match, no error
            continue
        elif tag == 'replace':
            # Words differ, check for errors
            for i in range(i1, i2):
                input_word = unicodedata.normalize('NFC', input_words[i])
                corrected_word = unicodedata.normalize('NFC', corrected_words[j1 + (i - i1)])
                print(f"Comparing replace at input[{i}]={input_word} -> corrected[{j1 + (i - i1)}]={corrected_word}")
                for dict_name, error_dict, error_type, reason in error_dictionaries:
                    if dict_name in ["indirect_object_errors", "possessive_noun_errors", "adverb_errors"]:
                        for correct_word, incorrect_word in error_dict.items():
                            if input_word == incorrect_word and corrected_word == correct_word:
                                error_key = f"{incorrect_word}_{correct_word}_{dict_name}_{i}"
                                if error_key not in seen_errors:
                                    explanation = {
                                        "incorrect": incorrect_word,
                                        "correct": correct_word,
                                        "error_type": error_type,
                                        "description": f"غلط لفظ '{incorrect_word}' استعمال ہوا، صحیح لفظ '{correct_word}' ہونا چاہیے۔",
                                        "reason": reason
                                    }
                                    detected_errors.append(explanation)
                                    seen_errors.add(error_key)
                                    print(f"Detected phrase error: {explanation}")
                    else:
                        for correct_word, incorrect_word in error_dict.items():
                            pattern = r'^' + re.escape(incorrect_word) + r'([ںےکیکوسےمیںوں]*)$'
                            input_match = re.match(pattern, input_word)
                            if input_match:
                                suffix = input_match.group(1)
                                if corrected_word == correct_word + suffix:
                                    error_key = f"{input_word}_{corrected_word}_{dict_name}_{i}"
                                    if error_key not in seen_errors:
                                        explanation = {
                                            "incorrect": input_word,
                                            "correct": corrected_word,
                                            "error_type": error_type,
                                            "description": f"غلط لفظ '{input_word}' استعمال ہوا، صحیح لفظ '{corrected_word}' ہونا چاہیے۔",
                                            "reason": reason
                                        }
                                        detected_errors.append(explanation)
                                        seen_errors.add(error_key)
                                        print(f"Detected error: {explanation}")
        elif tag == 'delete':
            # Input word was omitted, check if it was incorrect
            for i in range(i1, i2):
                input_word = unicodedata.normalize('NFC', input_words[i])
                print(f"Checking deleted input[{i}]={input_word}")
                for dict_name, error_dict, error_type, reason in error_dictionaries:
                    for correct_word, incorrect_word in error_dict.items():
                        pattern = r'^' + re.escape(incorrect_word) + r'([ںےکیکوسےمیںوں]*)$'
                        input_match = re.match(pattern, input_word)
                        if input_match:
                            suffix = input_match.group(1)
                            expected_correct = correct_word + suffix
                            error_key = f"{input_word}_{expected_correct}_{dict_name}_{i}"
                            if error_key not in seen_errors:
                                explanation = {
                                    "incorrect": input_word,
                                    "correct": expected_correct,
                                    "error_type": error_type,
                                    "description": f"غلط لفظ '{input_word}' استعمال ہوا، صحیح لفظ '{expected_correct}' ہونا چاہیے۔",
                                    "reason": reason
                                }
                                detected_errors.append(explanation)
                                seen_errors.add(error_key)
                                print(f"Detected omitted word error: {explanation}")
        elif tag == 'insert':
            # Corrected text added a word, skip for error detection
            print(f"Inserted words at corrected[{j1}:{j2}]={corrected_words[j1:j2]}")

    return detected_errors

class SentenceInput(BaseModel):
    input_text: str

@app.post("/mt5_paragraph")
async def mt5_paragraph(input_data: SentenceInput):
    input_text = input_data.input_text
    
    # Tokenize and correct the paragraph using the fine-tuned MT5 model
    prompt = "جملے کی درستگی: " + input_text
    inputs = mt5_paragraph_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = mt5_paragraph_model.generate(
            **inputs,
            max_length=512,
            num_beams=5,
            early_stopping=True
        )
    corrected_text = mt5_paragraph_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Detect errors by comparing input and corrected text
    errors = detect_errors(input_text, corrected_text)
    
    return {
        "input_text": input_text,
        "corrected_text": corrected_text,
        "errors": errors
    }

@app.post("/mt5_sentence")
async def mt5_sentence(input_data: SentenceInput):
    input_text = input_data.input_text
    
    # Tokenize and correct the sentence using the fine-tuned MT5 model
    prompt = "جملے کی درستگی: " + input_text
    inputs = mt5_sentence_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = mt5_sentence_model.generate(
            **inputs,
            max_length=512,
            num_beams=5,
            early_stopping=True
        )
    corrected_text = mt5_sentence_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Detect errors by comparing input and corrected text
    errors = detect_errors(input_text, corrected_text)
    
    return {
        "input_text": input_text,
        "corrected_text": corrected_text,
        "errors": errors
    }

@app.post("/bart_paragraph")
async def bart_paragraph(input_data: SentenceInput):
    input_text = input_data.input_text
    
    # Tokenize and correct the paragraph using the fine-tuned mBART model
    prompt = "جملے کی درستگی: " + input_text
    inputs = bart_paragraph_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs["decoder_input_ids"] = torch.tensor([[urdu_lang_id]])  # Set Urdu as target language
    with torch.no_grad():
        outputs = bart_paragraph_model.generate(
            **inputs,
            max_length=512,
            num_beams=5,
            early_stopping=True
        )
    corrected_text = bart_paragraph_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Detect errors by comparing input and corrected text
    errors = detect_errors(input_text, corrected_text)
    
    return {
        "input_text": input_text,
        "corrected_text": corrected_text,
        "errors": errors
    }
