#!/usr/bin/env python3

import json
import torch
import matplotlib.pyplot as plt

from helping_classes.dataset import Dataset
from helping_classes.model import Model
from helping_classes.trainer import Trainer

def main():
    # Initialize hyperparameters
    batch_size = 100
    learning_rate = 0.001
    num_epochs = 10

    # Create model
    model = Model()

    # Read json file with the separated data
    with open('json_files/dataset_files.json', 'r') as f:
        dataset_filenames = json.load(f)

    # Assign variables to the training and testing files
    train_filenames = dataset_filenames['training_files']
    test_filenames = dataset_filenames['testing_files']

    print('Used ' + str(len(train_filenames)) + ' for training and ' + str(len(test_filenames)) +
          ' for testing.')

    # Create objects from the Dataset Class using the new variables
    train_dataset = Dataset(train_filenames)
    test_dataset = Dataset(test_filenames)

    # Try the train_dataset
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # Train
    trainer = Trainer(model=model,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      learning_rate=learning_rate,
                      num_epochs=num_epochs,
                      model_path='models/model.pkl',
                      load_model=True)
    trainer.train()

    plt.show()

if __name__ == "__main__":
    main()
