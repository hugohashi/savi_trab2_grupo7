
#!/usr/bin/env python3

import torch
from torchvision import transforms
from PIL import Image


class Dataset(torch.utils.data.Dataset):

    def __init__(self, filenames):

        self.filenames = filenames
        self.number_of_images = len(self.filenames)

        # Compute the corresponding labels
        self.labels = []
        for filename in self.filenames:
            path_elements = filename.split('/')
            self.labels.append(path_elements[2])

        # Create a set of unique labels
        set_labels = set(self.labels)
        # print(set_labels)
        # print(len(set_labels))

        # Create a mapping from string labels to integer indexes
        self.label_to_index = {label: index for index, label in enumerate(set_labels)}
        # print(self.label_to_index)

        # Convert string labels to integer indexes
        self.int_labels = [self.label_to_index[label] for label in self.labels]
        # print(self.int_labels)

        # # Print the corresponding index for each filename
        # for filename, index in zip(self.filenames, self.int_labels):
        #     print(f"Filename: {filename}, Index: {index}")

        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])


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

        return tensor_image, label
    