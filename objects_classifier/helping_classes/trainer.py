#!/usr/bin/env python3

import os
from numpy import mean
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from colorama import Fore, Style

class Trainer():

    def __init__(self, model, train_loader, test_loader, learning_rate, num_epochs, model_path, load_model):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=learning_rate)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(Fore.BLUE + 'Device is ' + str(self.device) + Style.RESET_ALL)

        # Setup matplotlib figure
        plt.title('Training Objects Classifier', fontweight="bold")
        plt.axis([0, self.num_epochs, 0, 2])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        self.handle = None
        self.handle_test = None

        self.model_path = model_path

        self.load_model = load_model
        if not os.path.isfile(self.model_path):
            self.load_model = False

    def draw(self, epoch_train_losses, epoch_test_losses):
        xs = range(0, len(epoch_train_losses))
        ys = epoch_train_losses
        ys_test = epoch_test_losses

        if self.handle is None:  # draw first time
            self.handle = plt.plot(xs, ys, '-b')
            self.handle_test = plt.plot(xs, ys_test, '-r')
        else:  # edit plot all other times
            plt.setp(self.handle, xdata=xs, ydata=ys)
            plt.setp(self.handle_test, xdata=xs, ydata=ys_test)

        # Draw figure
        plt.draw()
        pressed_key = plt.waitforbuttonpress(0.1)
        if pressed_key == True:
            exit(0)

    def train(self):
        # Resume model or start from scratch
        if self.load_model:
            checkpoint = torch.load(self.model_path)

            self.model.load_state_dict(checkpoint['model_state_dict'])

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # https://github.com/pytorch/pytorch/issues/2830
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

            start_epoch = checkpoint['epoch_idx']
            epoch_train_losses = checkpoint['epoch_train_losses']
            epoch_test_losses = checkpoint['epoch_test_losses']

        else:
            start_epoch = 0
            epoch_train_losses = []
            epoch_test_losses = []

        self.model.to(self.device)

        for epoch_idx in range(start_epoch, self.num_epochs):
            print('Starting to train epoch ' + str(epoch_idx))

            # Train --------------------------------------------
            self.model.train()
            batch_losses = []
            for batch_idx, (inputs, labels_gt) in tqdm(enumerate(self.train_loader),
                                                       total=len(self.train_loader),
                                                       desc='Training batches for epoch ' + str(epoch_idx)):

                # move tensors to device
                inputs = inputs.to(self.device)
                labels_gt = labels_gt.clone().detach().to(self.device)

                # Get predicted labels
                labels_predicted = self.model.forward(inputs)

                # Compute loss comparing labels_predicted labels
                batch_loss = self.loss(labels_predicted, labels_gt)

                # Update model
                self.optimizer.zero_grad()  # resets the gradients from previous batches
                batch_loss.backward()
                self.optimizer.step()

                # Store batch loss
                # TODO should we not normalize the batch_loss by the number of images in the batch?
                batch_losses.append(batch_loss.data.item())

            # Compute epoch train loss
            epoch_train_loss = mean(batch_losses)
            epoch_train_losses.append(epoch_train_loss)

            # testing --------------------------------------------
            self.model.eval()
            batch_losses = []
            for batch_idx, (inputs, labels_gt) in tqdm(enumerate(self.test_loader),
                                                       total=len(self.test_loader),
                                                       desc='Validating batches for epoch ' + str(epoch_idx)):

                # move tensors to device
                inputs = inputs.to(self.device)
                labels_gt = labels_gt.clone().detach().to(self.device)

                # Get predicted labels
                labels_predicted = self.model.forward(inputs)

                # Compute loss comparing labels_predicted labels
                batch_loss = self.loss(labels_predicted, labels_gt)

                # Store batch loss
                # TODO should we not normalize the batch_loss by the number of images in the batch?
                batch_losses.append(batch_loss.data.item())

            # Compute epoch test loss
            epoch_test_loss = mean(batch_losses)
            epoch_test_losses.append(epoch_test_loss)

            print('Finished training epoch ' + str(epoch_idx))
            print('epoch_train_loss = ' + str(epoch_train_loss))
            print('epoch_test_loss = ' + str(epoch_test_loss))

            #Stop training the model if both losses are smaller than 0.1
            self.saveModel(model=self.model,
                        optimizer=self.optimizer,
                        epoch_idx=epoch_idx,
                        epoch_train_losses=epoch_train_losses,
                        epoch_test_losses=epoch_test_losses)
            print("model saved")

            self.draw(epoch_train_losses, epoch_test_losses)

            if epoch_train_loss < 0.1 and epoch_test_loss < 0.1:
                break


    def saveModel(self, model, optimizer, epoch_idx, epoch_train_losses, epoch_test_losses):
        print('Saving model to ' + self.model_path + ' ... ', end='')
        # Build a dictionary to save
        d = {'epoch_idx': epoch_idx,
             'model_state_dict': self.model.state_dict(),
             'optimizer_state_dict': self.optimizer.state_dict(),
             'epoch_train_losses': epoch_train_losses,
             'epoch_test_losses': epoch_test_losses}

        torch.save(d, self.model_path)

        print('Done.')