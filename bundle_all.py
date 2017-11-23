import argparse, subprocess, os

def main(args):

	layers = os.listdir(args.path)

	for layer in layers:

		p = os.path.join(args.path, layer)
		filters = os.listdir(p)

		for filter in filters:

			subprocess.call(["python", "bundle_images.py", os.path.join(p, filter)])

parser = argparse.ArgumentParser()

parser.add_argument("path")

parsed = parser.parse_args()
main(parsed)