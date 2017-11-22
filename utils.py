import cv2
import numpy as np
import tensorflow as tf

# synset = [l.strip() for l in open('synset.txt').readlines()]


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path):
    # load image
    img = cv2.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = cv2.resize(crop_img, (224, 224))
    return resized_img


# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1


def load_image2(path, height=None, width=None):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]
    return skimage.transform.resize(img, (ny, nx))


def test():
    img = skimage.io.imread("./test_data/starry_night.jpg")
    ny = 300
    nx = img.shape[1] * ny / img.shape[0]
    img = skimage.transform.resize(img, (ny, nx))
    skimage.io.imsave("./test_data/test/output.jpg", img)

def read_imgnet_labels(path):
      """
      Read and parse ILSVRC validation labels.
      :param path:    Path to the labels text fi;e/
      :return:        Parsed labels.
      """

      with open(path, "r") as f:
        content = f.readlines()

      content = [x.strip() for x in content]
      content = sorted(content)
      labels = [int(x.split(" ")[1]) for x in content]

      return labels

def argmax_2d(tensor):

      assert rank(tensor) == 4

      flat_tensor = tf.reshape(tensor, (tf.shape(tensor)[0], -1, tf.shape(tensor)[3]))
      argmax = tf.cast(tf.argmax(flat_tensor, axis=1), tf.int32)

      argmax_x = argmax // tf.shape(tensor)[2]
      argmax_y = argmax % tf.shape(tensor)[2]

      return tf.stack((argmax_x, argmax_y), axis=1)

def rank(tensor):
    return len(tensor.get_shape())

if __name__ == "__main__":
    test()
