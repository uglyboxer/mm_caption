import json
from os import listdir
from pathlib import Path
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model


def parse_cap_file(filepath, destination):
    with open(filepath) as f:
        js = json.load(f)

    captions = []

    for cap_dict in js['annotations']:
        image_id = '{0:012d}'.format(cap_dict['image_id']) 
        row = image_id + ' ' + cap_dict['caption']
        captions.append(row)

    with open(destination, 'w') as f:
        for r in captions:
            f.write(r)
            f.write('\n')


def parse_image_names(filepath, destination):
    image_ids = set()
    with open(filepath, 'r') as f:
        for line in f.readlines():
            id_ = line.split()
            image_ids.add(id_)

    with open(destination, 'w') as g:
        for x in image_ids:
            g.write(x)
            g.write('\n')
 
 
# extract features from each photo in the directory
def extract_features(directory):
    # load the model
    model = VGG16()
    # re-structure the model
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    # summarize
    print(model.summary())
    # extract features from each photo
    features = dict()
    for name in listdir(directory):
        # load an image from file
        filename = directory + '/' + name
        image = load_img(filename, target_size=(224, 224))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # get features
        feature = model.predict(image, verbose=0)
        # get image id
        image_id = name.split('.')[0]
        # store feature
        features[image_id] = feature
        print('>{}'.format(name))
    return features


# parse_cap_file('/Users/cole-home/data/coco/annotations/captions_train2017.json', '/Users/cole-home/data/coco/train_descriptions.txt')
# parse_cap_file('/Users/cole-home/data/coco/annotations/captions_val2017.json', '/Users/cole-home/data/coco/val_descriptions.txt')

parse_image_names('/Users/cole-home/data/coco/train_descriptions.txt', '/Users/cole-home/data/coco/train_images.txt')
parse_image_names('/Users/cole-home/data/coco/val_descriptions.txt', '/Users/cole-home/data/coco/val_images.txt')


# extract features from all images
home = str(Path.home())
directory = '{}/data/coco/train2017'.format(home)
features = extract_features(directory)
print('Extracted Features: {}'.format(len(features)))
# save to file
dump(features, open('{}/data/coco/train_features.pkl'.format(home), 'wb'))

directory = '{}/data/coco/val2017'.format(home)
features = extract_features(directory)
print('Extracted Features: {}'.format(len(features)))
# save to file
dump(features, open('{}/data/coco/val_features.pkl'.format(home), 'wb'))


