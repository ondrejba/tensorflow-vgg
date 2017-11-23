import argparse, cv2, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def three_grid(imgs, save_path):

	plt.figure(figsize = (3, 3), dpi=160)
	gs1 = gridspec.GridSpec(3, 3)
	gs1.update(wspace=0.025, hspace=0.05)

	for i in range(9):
	    ax1 = plt.subplot(gs1[i])
	    plt.axis('off')
	    ax1.set_aspect('equal')
	    plt.imshow(imgs[i])

	plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0, hspace=0)
	plt.savefig(save_path, transparent=True)

def two_grid(imgs, save_path):

	plt.figure(figsize = (2, 2))
	gs1 = gridspec.GridSpec(2, 2)
	gs1.update(wspace=0.0, hspace=0.0)

	for i in range(2):
	    ax1 = plt.subplot(gs1[i])
	    plt.axis('off')
	    ax1.set_aspect('equal')
	    plt.imshow(imgs[i])

	plt.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1, wspace=0, hspace=0)
	plt.savefig(save_path, transparent=True)


def bgr_to_rgb(img):
	tmp = img[:, :, 0].copy()
	img[:, :, 0] = img[:, :, 2]
	img[:, :, 2] = tmp
	return img

def main(args):
	
	deconv_paths = [os.path.join(args.path, "{}.jpg".format(i)) for i in range(9)]
	orig_paths = [os.path.join(args.path, "{}_orig.jpg".format(i)) for i in range(9)]

	for img_path in deconv_paths:
		assert os.path.isfile(img_path)

	for img_path in orig_paths:
		assert os.path.isfile(img_path)

	deconv_imgs = [bgr_to_rgb(cv2.imread(img_path) / 255.0) for img_path in deconv_paths]
	orig_imgs = [bgr_to_rgb(cv2.imread(img_path) / 255.0) for img_path in orig_paths]

	deconv_save = os.path.join(args.path, "deconv.png")
	orig_save = os.path.join(args.path, "orig.png")

	three_grid(orig_imgs, orig_save)
	three_grid(deconv_imgs, deconv_save)

	#two_imgs = [bgr_to_rgb(cv2.imread(img_path) / 255.0) for img_path in [deconv_save, orig_save]]

	#full_save = os.path.join(args.path, "full.png")
	#two_grid(two_imgs, full_save)

parser = argparse.ArgumentParser()

parser.add_argument("path")

parsed = parser.parse_args()
main(parsed)