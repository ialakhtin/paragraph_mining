import nltk
nltk.download('punkt_tab')      
nltk.download('wordnet')    
nltk.download('omw-1.4') 
nltk.download('averaged_perceptron_tagger_eng')

import os
import json
import pymorphy3
import argostranslate.translate
from tqdm import tqdm
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

data_dir = 'data/validator'
ru_lemmas_path = 'ru_w2l.json'
en_lemmas_path = 'en_w2l.json'

print("Russain")
morph = pymorphy3.MorphAnalyzer()
with open(os.path.join(data_dir, 'russian_words.txt'), encoding='windows-1251') as f:
    ru_words = f.readlines()
    ru_words = [word.lower().strip() for word in ru_words]
ru_lemmas = [morph.parse(word)[0].normal_form for word in tqdm(ru_words)]
ru_word_to_lemma = {w: l for w, l in zip(ru_words, ru_lemmas)}
with open(os.path.join(data_dir, ru_lemmas_path), "w", encoding='utf-8') as fp:
    json.dump(ru_word_to_lemma, fp, ensure_ascii=False)
ru_lemmas = list(set(ru_lemmas))
print(f"{len(ru_lemmas)} ru lemmas")

print("English")
lemmatizer = WordNetLemmatizer()
with open(os.path.join(data_dir, 'english_words.txt')) as f:
    en_words = f.readlines()
    en_words = [word.lower().strip() for word in en_words]
en_lemmas = [
    lemmatizer.lemmatize(i, j[0].lower()) 
    if j[0].lower() in ['a','n','v'] else 
    lemmatizer.lemmatize(i) 
    for i, j in pos_tag(en_words)
]
en_word_to_lemma = {w: l for w, l in zip(en_words, en_lemmas)}
with open(os.path.join(data_dir, en_lemmas_path), "w", encoding='utf-8') as fp:
    json.dump(en_word_to_lemma, fp, ensure_ascii=False)
en_lemmas = list(set(en_lemmas))
print(f"{len(en_lemmas)} en lemmas")

print("Translation")

argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()

def translate(lang_src, lang_dst, lemmas_src, dict_dst):
    package_to_install = next(
        filter(
            lambda x: x.from_code == lang_src and x.to_code == lang_dst, available_packages
        )
    )
    argostranslate.package.install_from_path(package_to_install.download())

    translated = [argostranslate.translate.translate(t, lang_src, lang_dst) for t in tqdm(lemmas_src)]
    pairs = []
    for s, d in zip(lemmas_src, translated):
        if d not in dict_dst:
            continue
        d = dict_dst[d]
        pairs.append((s, d))

    translator_d = {s: d for s, d in pairs}
    with open(os.path.join(data_dir, f'{lang_src}_{lang_dst}.json'), "w", encoding='utf-8') as fp:
        json.dump(translator_d, fp, ensure_ascii=False)

translate('ru', 'en', ru_lemmas, en_word_to_lemma)
translate('en', 'ru', en_lemmas, ru_word_to_lemma)