import os
from PIL import Image
import json

# We use a subset of the labels
desired_labels = ['car','truck','bus','motorcycle','bicycle', 'traffic sign','traffic light', 'vegetation']
target_path = './data/results/'

# Transforms cityscapes data for class instance segmentation into single image classification task
# It uses polygon data and calculates the smallest rectangle around it
# More advanced methods e.g. using box overlap might improve the quality of the data
def generate_classifications():
    # What minimal size (in pixels) should the results have
    # This is an arbitrary constant
    minimal_dimensions = 64

    # Classes are strongly imbalanced, we limit the number of instances for the biggest classes
    limit_per_label = 1000

    label_counts = {label: 0 for label in desired_labels}

    path_base_jsons = './data/gtFine/train/'
    path_base_images = './data/leftImg8bit_trainvaltest/leftImg8bit/train/'

    # prepare the folders
    for label in desired_labels:
        os.makedirs(os.path.join(target_path + label), exist_ok=True)

    id = 0
    for subdir, dirs, files in os.walk(path_base_jsons):
        for f in files:
            if not f.endswith('.json'):
                continue

            path_image = os.path.join(
                path_base_images,
                os.path.basename(subdir),
                f[:-21] + '_leftImg8bit.png'
            )

            try:
                img = Image.open(path_image)
            except:
                print('invalid path: ' + path_image)

            history = {}
            with open(subdir + '/' + f) as polygons:
                d = json.load(polygons)

                for o in d['objects']:
                    label = o['label']
                    if label not in desired_labels or label_counts[label] == limit_per_label:
                        continue

                    # Find the bounding rectangle
                    x1 = 100000
                    x2 = 0
                    y1 = 100000
                    y2 = 0

                    for coord in o['polygon']:
                        if coord[0] < x1:
                            x1 = coord[0]
                        if coord[0] > x2:
                            x2 = coord[0]
                        if coord[1] < y1:
                            y1 = coord[1]
                        if coord[1] > y2:
                            y2 = coord[1]

                    if x2 - x1 < minimal_dimensions or y2 - y1 < minimal_dimensions:
                        continue

                    # Ignore duplicates
                    if (x1, x2) in history.values():
                        continue

                    label_counts[label] += 1

                    history[id] = (x1, x2)

                    cropped_img = img.crop((x1, y1, x2, y2))

                    cropped_img.save(os.path.join('./data/results/', label, str(id) + '.png'))
                    id += 1

generate_classifications()

# Print final counts
for label in desired_labels:
    path = os.path.join(target_path + label)
    print(label + ': ' + str(len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])))
