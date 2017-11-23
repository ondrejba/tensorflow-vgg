import cv2, os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

TEST_DECONV_IMGS = ["data_backup_2/s0/data/run58/layer0/filter0/{}.jpg".format(idx) for idx in range(9)]
TEST_ORIG_IMGS = ["data_backup_2/s0/data/run58/layer0/filter0/{}_orig.jpg".format(idx) for idx in range(9)]

for img_path in TEST_DECONV_IMGS:
	assert os.path.isfile(img_path)

for img_path in TEST_ORIG_IMGS:
	assert os.path.isfile(img_path)

DECONV_IMGS = [cv2.imread(img_path) / 255 for img_path in TEST_DECONV_IMGS]
ORIG_IMGS = [cv2.imread(img_path) / 255 for img_path in TEST_ORIG_IMGS]

#for i in range(0, 9):
#    plt.subplot(331 + (i))
#    plt.axis("off")
#    plt.imshow(DECONV_IMGS[i])

#plt.show()

plt.figure(figsize = (3, 3))
gs1 = gridspec.GridSpec(3, 3)
gs1.update(wspace=0.025, hspace=0.05)

for i in range(9):
   # i = i + 1 # grid spec indexes from 0
    ax1 = plt.subplot(gs1[i])
    ax1.set_aspect('equal')
    plt.axis("off")
    ax1.imshow(DECONV_IMGS[i])

plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0, hspace=0)
plt.savefig("demo.png", transparent=True)