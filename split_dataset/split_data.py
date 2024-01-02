
#!/usr/bin/env python3

import glob
import json
from sklearn.model_selection import train_test_split
import os


def main():
    data_path = '../data/objects/rgbd-dataset'
    data_files = glob.glob(os.path.join(data_path, '**/*_crop.png'), recursive=True)

    # Get the set of items
    
    # objects = []
    # for file in data_files:
    #     path_elements = file.split('/')

    #     # Get the name of the object
    #     object_name = path_elements[4]

    #     objects.append(object_name)

    # set_objects = set(objects)
    # print(set_objects)

    training_files, testing_files = train_test_split(data_files, train_size=0.8, test_size=0.2)

    print('We have a total of ' + str(len(data_files)) + ' images.')
    print('Used ' + str(len(training_files)) + ' as training images')
    print('Used ' + str(len(testing_files)) + ' as testing images')

    d = {'training_files': training_files,
         'testing_files': testing_files}

    json_object = json.dumps(d, indent=2)

    # Writing to sample.json
    with open("dataset_files.json", "w") as outfile:
        outfile.write(json_object)


if __name__ == "__main__":
    main()
