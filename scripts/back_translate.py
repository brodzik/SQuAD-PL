import glob
import json
import os
import re
import sys
import time

from deep_translator import GoogleTranslator
from joblib import Parallel, delayed
from tqdm import tqdm


def get_qas():
    with open("train-v2.0.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    for group_id, group in enumerate(data["data"]):
        for passage_id, passage in enumerate(group["paragraphs"]):
            context = passage["context"]
            for qa in passage["qas"]:
                question = qa["question"]

                if len(qa["answers"]) == 0:
                    yield context, question, {"text": "", "answer_start": 0}, group_id, passage_id

                for answer in qa["answers"]:
                    yield context, question, answer, group_id, passage_id


def split_context(context):
    context_parts = []
    current = ""

    for token in context.split(" "):
        if len(current) + len(token) + 1 < 5000:
            current += " " + token
        else:
            context_parts.append(current.strip())
            current = ""

    if len(current) > 0:
        context_parts.append(current.strip())

    return context_parts


def preprocess(qa_id, row):
    try:
        _preprocess(qa_id, row)
    except:
        pass


translator1 = GoogleTranslator(source="en", target="ru")
translator2 = GoogleTranslator(source="ru", target="pl")


def translate(text):
    return translator2.translate(translator1.translate(text))


context_translated_cache = {}


def _preprocess(qa_id, row):
    global context_translated_cache

    context = str(row[0])
    question = str(row[1])
    answer_text = str(row[2]["text"])
    answer_start = int(row[2]["answer_start"])
    group_id = int(row[3])
    passage_id = int(row[4])

    if answer_text and not answer_text.isdigit():
        answer_text_translated = translate(answer_text)
        time.sleep(0.5)
    else:
        answer_text_translated = answer_text

    context_id = str(group_id) + str(passage_id)
    if context_id in context_translated_cache:
        context_translated = context_translated_cache[context_id]
    else:
        context_translated = " ".join([translate(c) for c in split_context(context)])
        time.sleep(0.5)
        context_translated_cache[context_id] = context_translated

    if answer_text_translated:
        if answer_text_translated not in context_translated:
            return

        start_candidates = [m.start() for m in re.finditer(answer_text_translated, context_translated)]
        answer_start_translated = min(start_candidates, key=lambda x: abs(x - answer_start))
    else:
        answer_start_translated = 0

    question_translated = translate(question)
    time.sleep(0.5)

    with open("out/qa_{}.json".format(qa_id), "w", encoding="utf-8") as f:
        json.dump({
            "context": context_translated,
            "question": question_translated,
            "answer_text": answer_text_translated,
            "answer_start": answer_start_translated,
            "group_id": group_id,
            "passage_id": passage_id
        }, f)


if __name__ == "__main__":
    os.makedirs("out", exist_ok=True)
    sys.stdout.reconfigure(encoding="utf-8")

    Parallel(n_jobs=6, backend="multiprocessing")(delayed(preprocess)(qa_id, row) for qa_id, row in enumerate(tqdm(get_qas(), total=130319)))
