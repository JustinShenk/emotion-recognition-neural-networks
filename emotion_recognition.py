#!/usr/bin/python3
import numpy as np
import mvnc.mvncapi as mvnc
import skimage
from skimage import io, transform
import numpy
import os
import sys

# User modifiable input parameters
NCAPPZOO_PATH = './'
GRAPH_PATH = NCAPPZOO_PATH + 'gpu_model/graph2'
IMAGE_PATH = NCAPPZOO_PATH + 'data/images/face.jpg'
LABELS_FILE_PATH = NCAPPZOO_PATH + 'data/fer2013/emotions.txt'
IMAGE_MEAN = 0.55
IMAGE_STDDEV = 0.248
IMAGE_DIM = (48, 48)


class EmotionRecognition:
    def __init__(self):
        # Look for enumerated NCS device(s); quit program if none found.
        devices = mvnc.enumerate_devices()
        if len(devices) == 0:
            print('No devices found')
            quit()

        # Get a handle to the first enumerated device and open it
        self.device = mvnc.Device(devices[0])
        self.device.open()

        # Load a graph file onto the NCS device

        # Read the graph file into a buffer
        with open(GRAPH_PATH, mode='rb') as f:
            blob = f.read()

        # Load the graph buffer into the NCS
        self.graph = mvnc.Graph('graph')
        self.fifoIn, self.fifoOut = self.graph.allocate_with_fifos(self.device, blob)

    def __exit__(self, exc_type, exc_value, traceback):
        """Unload the graph and close the device. """
        self.fifoIn.destroy()
        self.fifoOut.destroy()
        self.graph.destroy()
        self.device.close()

    def predict(self, face=None, image_path=None, normalize=False):
        """ Turn image or image_path into a prediction.
        Args:
            face        48x48 grayscale face image scaled from 0 to 1
            image_path  Path to image
        Returns:
            result  score, eg., [[0.11 0.0415 ...]]

        """
        # Read & resize image [Image size is defined during training]
        if image_path is not None:
            print("Loading image from {}".format(image_path))
            img = skimage.io.imread(image_path, as_grey=True)
            img = skimage.transform.resize(
                img, IMAGE_DIM, preserve_range=True)
        else:
            img = face
        if img is None:
            print("No image found")
            return None
        if normalize:
            img = (img - IMAGE_MEAN) * IMAGE_STDDEV

        # Load the image as a half-precision floating point array
        self.graph.queue_inference_with_fifo_elem(self.fifoIn, self.fifoOut, img.astype(np.float32), 'user object')
        output, userobj = self.fifoOut.read_elem()

        # Get the results from NCS

        # Print the results
        print('\n------- predictions --------')

        labels = numpy.loadtxt(LABELS_FILE_PATH, str, delimiter='\t')

        order = output.argsort()[::-1][:6]

        for i in range(0, 4):
            print('prediction ' + str(i) + ' is ' + labels[order[i]])

        # If a display is available, show the image on which inference was performed
        if 'DISPLAY' in os.environ:
            if image_path is not None:
                skimage.io.imshow(IMAGE_PATH)
                skimage.io.show()
            else:
                skimage.io.imshow(face)

        return [output]
