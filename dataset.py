
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
            # basename = os.path.basename(filename)
            path_elements = filename.split('/')
            label = path_elements[4]  # because basename is "cat.2109.jpg"

            self.labels.append(label)

            # if label == 'dog':
            #     self.labels.append(0)
            # elif label == 'cat':
            #     self.labels.append(1)
            # else:
            #     raise ValueError('Unknown label ' + label)

        self.set_labels = set(self.labels)

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
