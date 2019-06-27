"""Runs BERT over input data and writes out its attention maps to disk."""

import argparse
import os
import numpy as np
import tensorflow as tf

from bert import modeling
from bert import tokenization
import bpe_utils
import utils


class Example(object):
  """Represents a single input sequence to be passed into BERT."""

  def __init__(self, features, tokenizer, max_sequence_length,):
    self.features = features

    if "tokens" in features:
      self.tokens = features["tokens"]
    else:
      if "text" in features:
        text = features["text"]
      else:
        text = " ".join(features["words"])
      self.tokens = ["[CLS]"] + tokenizer.tokenize(text) + ["[SEP]"]

    self.input_ids = tokenizer.convert_tokens_to_ids(self.tokens)
    self.segment_ids = [0] * len(self.tokens)
    self.input_mask = [1] * len(self.tokens)
    while len(self.input_ids) < max_sequence_length:
      self.input_ids.append(0)
      self.input_mask.append(0)
      self.segment_ids.append(0)


def examples_in_batches(examples, batch_size):
  for i in utils.logged_loop(range(1 + ((len(examples) - 1) // batch_size))):
    yield examples[i * batch_size:(i + 1) * batch_size]


class AttnMapExtractor(object):
  """Runs BERT over examples to get its attention maps."""

  def __init__(self, bert_config_file, init_checkpoint,
               max_sequence_length=128, debug=False):
    make_placeholder = lambda name: tf.placeholder(
        tf.int32, shape=[None, max_sequence_length], name=name)
    self._input_ids = make_placeholder("input_ids")
    self._segment_ids = make_placeholder("segment_ids")
    self._input_mask = make_placeholder("input_mask")

    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    if debug:
      bert_config.num_hidden_layers = 3
      bert_config.hidden_size = 144
    self._attn_maps = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=self._input_ids,
        input_mask=self._input_mask,
        token_type_ids=self._segment_ids,
        use_one_hot_embeddings=True).attn_maps

    if not debug:
      print("Loading BERT from checkpoint...")
      assignment_map, _ = modeling.get_assignment_map_from_checkpoint(
          tf.trainable_variables(), init_checkpoint)
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

  def get_attn_maps(self, sess, examples):
    feed = {
        self._input_ids: np.vstack([e.input_ids for e in examples]),
        self._segment_ids: np.vstack([e.segment_ids for e in examples]),
        self._input_mask: np.vstack([e.input_mask for e in examples])
    }
    return sess.run(self._attn_maps, feed_dict=feed)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--preprocessed-data-file", required=True,
      help="Location of preprocessed data (JSON file); see the README for "
           "expected data format.")
  parser.add_argument("--bert-dir", required=True,
                      help="Location of the pre-trained BERT model.")
  parser.add_argument("--cased", default=False, action='store_true',
                      help="Don't lowercase the input.")
  parser.add_argument("--max_sequence_length", default=128, type=int,
                      help="Maximum input sequence length after tokenization "
                           "(default=128).")
  parser.add_argument("--batch_size", default=16, type=int,
                      help="Batch size when running BERT (default=16).")
  parser.add_argument("--debug", default=False, action='store_true',
                      help="Use tiny model for fast debugging.")
  parser.add_argument("--word_level", default=False, action='store_true',
                      help="Get word-level rather than token-level attention.")
  args = parser.parse_args()

  print("Creating examples...")
  tokenizer = tokenization.FullTokenizer(
      vocab_file=os.path.join(args.bert_dir, "vocab.txt"),
      do_lower_case=not args.cased)
  examples = []
  for features in utils.load_json(args.preprocessed_data_file):
    example = Example(features, tokenizer, args.max_sequence_length)
    if len(example.input_ids) <= args.max_sequence_length:
      examples.append(example)

  print("Building BERT model...")
  extractor = AttnMapExtractor(
      os.path.join(args.bert_dir, "bert_config.json"),
      os.path.join(args.bert_dir, "bert_model.ckpt"),
      args.max_sequence_length, args.debug
  )

  print("Extracting attention maps...")
  feature_dicts_with_attn = []
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for batch_of_examples in examples_in_batches(examples, args.batch_size):
      attns = extractor.get_attn_maps(sess, batch_of_examples)
      for e, e_attn in zip(batch_of_examples, attns):
        seq_len = len(e.tokens)
        e.features["attns"] = e_attn[:, :, :seq_len, :seq_len].astype("float16")
        e.features["tokens"] = e.tokens
        feature_dicts_with_attn.append(e.features)

  if args.word_level:
    print("Converting to word-level attention...")
    bpe_utils.make_attn_word_level(
        feature_dicts_with_attn, tokenizer, args.cased)

  outpath = args.preprocessed_data_file.replace(".json", "")
  outpath += "_attn.pkl"
  print("Writing attention maps to {:}...".format(outpath))
  utils.write_pickle(feature_dicts_with_attn, outpath)
  print("Done!")


if __name__ == "__main__":
  main()