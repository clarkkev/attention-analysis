"""Does BERT-style preprocessing of unlabeled data; heavily based on
create_pretraining_data.py in the BERT codebase. However, does not mask words
or ever use random paragraphs for the second text segment."""

import argparse
import os
import random

import utils
from bert import tokenization


def prep_document(document, max_sequence_length):
  """Does BERT-style preprocessing on the provided document."""
  max_num_tokens = max_sequence_length - 3
  target_seq_length = max_num_tokens

  # We DON"T just concatenate all of the tokens from a document into a long
  # sequence and choose an arbitrary split point because this would make the
  # next sentence prediction task too easy. Instead, we split the input into
  # segments "A" and "B" based on the actual "sentences" provided by the user
  # input.
  instances = []
  current_chunk = []
  current_length = 0
  i = 0
  while i < len(document):
    segment = document[i]
    current_chunk.append(segment)
    current_length += len(segment)
    if i == len(document) - 1 or current_length >= target_seq_length:
      if current_chunk:
        a_end = 1
        if len(current_chunk) >= 2:
          a_end = random.randint(1, len(current_chunk) - 1)

        tokens_a = []
        for j in range(a_end):
          tokens_a.extend(current_chunk[j])

        tokens_b = []
        for j in range(a_end, len(current_chunk)):
          tokens_b.extend(current_chunk[j])
        truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, random)

        if len(tokens_a) == 0 or len(tokens_b) == 0:
          break
        assert len(tokens_a) >= 1
        assert len(tokens_b) >= 1

        tokens = []
        tokens.append("[CLS]")
        for token in tokens_a:
          tokens.append(token)

        tokens.append("[SEP]")

        for token in tokens_b:
          tokens.append(token)
        tokens.append("[SEP]")

        instances.append(tokens)

      current_chunk = []
      current_length = 0
    i += 1

  return instances


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
  """Truncates a pair of sequences to a maximum sequence length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
    assert len(trunc_tokens) >= 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    if rng.random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--data-file", required=True,
      help="Location of input data; see the README for expected data format.")
  parser.add_argument("--bert-dir", required=True,
                      help="Location of the pre-trained BERT model.")
  parser.add_argument("--num-docs", default=1000, type=int,
                      help="Number of documents to use (default=1000).")
  parser.add_argument("--cased", default=False, action='store_true',
                      help="Don't lowercase the input.")
  parser.add_argument("--max_sequence_length", default=128, type=int,
                      help="Maximum input sequence length after tokenization "
                           "(default=128).")

  args = parser.parse_args()

  random.seed(0)
  current_doc_tokens = []
  segments = []
  tokenizer = tokenization.FullTokenizer(
      vocab_file=os.path.join(args.bert_dir, "vocab.txt"),
      do_lower_case=not args.cased)

  with open(args.data_file, "r") as f:
    for line in f:
      line = tokenization.convert_to_unicode(line).strip()

      # Empty lines are used as document delimiters
      if not line:
        if current_doc_tokens:
          for segment in prep_document(
              current_doc_tokens, args.max_sequence_length):
            segments.append(segment)
            if len(segments) >= args.num_docs:
              break
          if len(segments) >= args.num_docs:
            break
        current_doc_tokens = []
      tokens = tokenizer.tokenize(line)
      if tokens:
        current_doc_tokens.append(tokens)

  # # Remove empty documents
  # all_documents = [x for x in all_documents if x]
  # random.shuffle(all_documents)
  #
  # tokens = []
  # print("Preprocessing data...")
  # for doc in all_documents:
  #   tokens += prep_document(doc, args.max_sequence_length)
  #
  # random.shuffle(tokens)
  # tokens = tokens[:args.num_docs]

  random.shuffle(segments)
  utils.write_json([{"tokens": s} for s in segments],
                   args.data_file.replace(".txt", "") + ".json")


if __name__ == "__main__":
  main()
