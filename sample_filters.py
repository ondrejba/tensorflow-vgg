import argparse, os, json, random

def main(args):

    layers = {}
    num_filters = [64, 64, 64,
                   128, 128, 128,
                   256, 256, 256, 256,
                   512, 512, 512, 512,
                   512, 512, 512, 512]

    for layer_idx in range(16):

        layer_num_filters = num_filters[layer_idx]

        filter_idxs = set()

        while len(filter_idxs) != args.num_per_layer:

            filter_idxs.add(random.randint(0, layer_num_filters - 1))

        filter_idxs = list(sorted(list(filter_idxs)))
        layers[layer_idx] = filter_idxs

    dir = "resources"

    if not os.path.isdir(dir):
      os.makedirs(dir)

    path = os.path.join(dir, "filters.json")

    with open(path, "w") as file:
      json.dump(layers, file, sort_keys=True, indent=4, separators=(',', ': '))

parser = argparse.ArgumentParser()

parser.add_argument("num_per_layer", type=int)

parsed = parser.parse_args()
main(parsed)