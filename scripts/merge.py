import pandas as pd
import argparse
import sys
import glob
from tqdm.auto import tqdm
import json


def main(args):
    df = []

    for filepath in tqdm(glob.glob(args.input_directory + "/*.json")):
        qa_id = int(filepath.split("qa_")[1].split(".json")[0])

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            data["qa_id"] = qa_id

        df.append(data)

    df = pd.DataFrame(df)
    df.sort_values(by="qa_id", inplace=True, ignore_index=True)
    df = df[["qa_id", "group_id", "passage_id", "context", "question", "answer_text", "answer_start"]]
    df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_directory")
    parser.add_argument("output_file")
    args = parser.parse_args()

    main(args)
