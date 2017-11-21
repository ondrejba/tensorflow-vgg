import argparse, os, subprocess

WORKER_ADDRESSES = [
  "root@80.188.166.38",
  "root@gpu-dlab01.de.showmax.cc",
  "root@gpu-dlab02.de.showmax.cc",
  "root@gpu-dlab03.de.showmax.cc",
  "root@gpu-dlab04.de.showmax.cc",
  "root@gpu-dlab05.de.showmax.cc",
  "root@gpu-dlab06.de.showmax.cc",
  "root@gpu-dlab07.de.showmax.cc",
  "root@gpu-dlab08.de.showmax.cc"
]

PATHS = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
LOCAL_SUMMARY_PATH = "data"
REMOTE_SUMMARY_PATH = "projects/tensorflow-vgg/data"

def download_for_worker(worker_idx):

  address = WORKER_ADDRESSES[worker_idx]

  path = PATHS[worker_idx]
  path = os.path.join(LOCAL_SUMMARY_PATH, path)

  if not os.path.isdir(path):
    os.makedirs(path)

  subprocess.call(["rsync", "-avz", "-e", "ssh", "{:s}:{:s}".format(address, REMOTE_SUMMARY_PATH),
                   "{:s}".format(path)])

def main(args):

  if args.worker is not None:
    assert 0 <= args.worker <= 8
    download_for_worker(args.worker)
  elif args.all is not None:
    for worker_idx, worker in enumerate(WORKER_ADDRESSES):
      download_for_worker(worker_idx)
  else:
    print("No action.")

parser = argparse.ArgumentParser()
parser.add_argument("-w", "--worker", type=int)
parser.add_argument("-a", "--all", default=False, action="store_true")
parsed = parser.parse_args()
main(parsed)