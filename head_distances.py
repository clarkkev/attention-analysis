"""Computes the average Jenson-Shannon Divergence between attention heads."""

import argparse
import numpy as np

import utils


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--attn-data-file", required=True,
    help="Pickle file containing extracted attention maps.")
  parser.add_argument("--outfile", required=True,
                      help="Where to write out the distances between heads.")
  args = parser.parse_args()

  print("Loading attention data")
  data = utils.load_pickle(args.attn_data_file)

  print("Computing head distances")
  js_distances = np.zeros([144, 144])
  for doc in utils.logged_loop(data, n_steps=None):
    if "attns" not in doc:
      continue
    tokens, attns = doc["tokens"], np.array(doc["attns"])

    attns_flat = attns.reshape([144, attns.shape[2], attns.shape[3]])
    for head in range(144):
      head_attns = np.expand_dims(attns_flat[head], 0)
      head_attns_smoothed = (0.001 / head_attns.shape[1]) + (head_attns * 0.999)
      attns_flat_smoothed = (0.001 / attns_flat.shape[1]) + (attns_flat * 0.999)
      m = (head_attns_smoothed + attns_flat_smoothed) / 2
      js = -head_attns_smoothed * np.log(m / head_attns_smoothed)
      js += -attns_flat_smoothed * np.log(m / attns_flat_smoothed)
      js /= 2
      js = js.sum(-1).sum(-1)
      js_distances[head] += js

    utils.write_pickle(js_distances, args.outfile)


if __name__ == "__main__":
  main()
