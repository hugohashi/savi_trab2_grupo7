#!/usr/bin/env python3

import glob
import json
from sklearn.model_selection import train_test_split
import os

def main():
    # Read data
    data_path = '../data/rgbd-dataset'
    data_files = glob.glob(os.path.join(data_path, '**/*_crop.png'), recursive=True)

    # Separate testing and training data into 80%/20%
    training_files, testing_files = train_test_split(data_files, train_size=0.8, test_size=0.2)

    print('We have a total of ' + str(len(data_files)) + ' images.')
    print('Used ' + str(len(training_files)) + ' as training images')
    print('Used ' + str(len(testing_files)) + ' as testing images')

    # Create a dictionary with the separated data
    d = {'training_files': training_files,
         'testing_files': testing_files}

    json_object = json.dumps(d, indent=2)

    # Write the dictionary to a json file
    with open("json_files/dataset_files.json", "w") as outfile:
        outfile.write(json_object)

if __name__ == "__main__":
    main()
