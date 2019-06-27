"""Preprocesses dependency parsing data and writes the result as JSON."""

import argparse
import os

import utils


def preprocess_depparse_data(raw_data_file):
  examples = []
  with open(raw_data_file) as f:
    current_example = {"words": [], "relns": [], "heads": []}
    for line in f:
      line = line.strip()
      if line:
        word, label = line.split()
        head, reln = label.split("-")
        head = int(head)
        current_example["words"].append(word)
        current_example["relns"].append(reln)
        current_example["heads"].append(head)
      else:
        examples.append(current_example)
        current_example = {"words": [], "relns": [], "heads": []}
  utils.write_json(examples, raw_data_file.replace(".txt", ".json"))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--data-dir", required=True,
      help="The location of dependency parsing data. Should contain files "
           "train.txt and dev.txt. See the README for expected data format.")
  args = parser.parse_args()
  for split in ["train", "dev"]:
    print("Preprocessing {:} data...".format(split))
    preprocess_depparse_data(os.path.join(args.data_dir, split + ".txt"))
  print("Done!")


if __name__ == "__main__":
  main()
