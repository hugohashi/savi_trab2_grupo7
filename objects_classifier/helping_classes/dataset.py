#!/usr/bin/env python3

from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms
from PIL import Image
import json
import os

class Dataset(TorchDataset):
    label_mapping_file = "json_files/label_mapping.json"
    label_mapping = {}

    def __init__(self, filenames):
        self.filenames = filenames
        self.number_of_images = len(self.filenames)
        self.load_label_mapping()

        # Compute the labels
        self.labels = []
        for filename in self.filenames:
            path_elements = filename.split('/')
            label = path_elements[3]

            if label not in self.label_mapping:
                # Assign a new numerical value for the unseen label
                self.label_mapping[label] = len(self.label_mapping)

            self.labels.append(self.label_mapping[label])

        # print(self.label_mapping)

        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.save_label_mapping()

    def __len__(self):
        # Returns the size of the data
        return self.number_of_images

    def __getitem__(self, index):
        filename = self.filenames[index]
        pil_image = Image.open(filename)

        # Convert to tensor
        tensor_image = self.transforms(pil_image)

        # Get corresponding label
        label = self.labels[index]
        return tensor_image, label, filename

    def save_label_mapping(self):
        # Save the labels/indexes in a dictionary
        with open(self.label_mapping_file, 'w') as file:
            json.dump(self.label_mapping, file)

    def load_label_mapping(self):
        # If the labels/indexes dictionary exists load it 
        if os.path.exists(self.label_mapping_file):
            with open(self.label_mapping_file, 'r') as file:
                self.label_mapping = json.load(file)
        else:
            self.label_mapping = {}
