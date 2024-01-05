
#!/usr/bin/env python3

import json
import torch
import matplotlib.pyplot as plt

from dataset import Dataset
from model import Model
from trainer import Trainer


def main():

    # Initialize hyperparameters
    batch_size = 100
    learning_rate = 0.001
    num_epochs = 30

    # Create model
    model = Model()

    # Prepare Datasets
    with open('split_dataset/dataset_files.json', 'r') as f:
        # Read json file
        dataset_filenames = json.load(f)

    train_filenames = dataset_filenames['training_files']
    testing_filenames = dataset_filenames['testing_files']

    print('Used ' + str(len(train_filenames)) + ' for training and ' + str(len(testing_filenames)) +
          ' for testing.')

    train_dataset = Dataset(train_filenames)
    testing_dataset = Dataset(testing_filenames)

    # Try the train_dataset
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset=testing_dataset, batch_size=batch_size, shuffle=True)

    # Train
    trainer = Trainer(model=model,
                      train_loader=train_loader,
                      validation_loader=validation_loader,
                      learning_rate=learning_rate,
                      num_epochs=num_epochs,
                      model_path='models/checkpoint.pkl',
                      load_model=True,
                      label_to_index=train_dataset.label_to_index)
    trainer.train()

    plt.show()

if __name__ == "__main__":
    main()
