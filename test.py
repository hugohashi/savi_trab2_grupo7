
#!/usr/bin/env python3

import json
import torch
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from dataset import Dataset
from model import Model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('split_dataset/dataset_files.json', 'r') as f:
    dataset_filenames = json.load(f)

test_filenames = dataset_filenames['testing_files'][:100]

test_dataset = Dataset(test_filenames)

print(f'Used {len(test_filenames)} for testing ')

with open('label_mapping.json', 'r') as f:
    label_mapping = json.load(f)


def get_key_by_value(dictionary, target_value):
    for key, value in dictionary.items():
        if value == target_value:
            return key
    # If the value is not found, you can return a default value or raise an exception
    return None

def evaluate_model(model, test_loader, device):
    model.eval()

    all_predicted_labels = []
    all_ground_truth_labels = []

    with torch.no_grad():
        for inputs, labels_gt in test_loader:
            inputs = inputs.to(device)
            labels_gt = labels_gt.clone().detach().to(device)

            labels_predicted = model(inputs)
            predicted_probabilities = F.softmax(labels_predicted, dim=1)
            predicted_indexes = torch.argmax(predicted_probabilities, dim=1).cpu().numpy()

            all_predicted_labels.extend(predicted_indexes)
            all_ground_truth_labels.extend(labels_gt.cpu().numpy())

    confusion_mat = confusion_matrix(all_ground_truth_labels, all_predicted_labels)
    TP, FP, FN, TN = confusion_mat[1, 1], confusion_mat[0, 1], confusion_mat[1, 0], confusion_mat[0, 0]

    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

    return precision, recall, f1_score, TP, TN, FP, FN

def display_images_with_labels(inputs, ground_truth_labels, predicted_labels):
    tensor_to_pil_image = transforms.ToPILImage()

    fig = plt.figure(figsize=(12, 12))
    for idx_image in range(len(ground_truth_labels)):
        image_tensor = inputs[idx_image, :, :, :]
        image_pil = tensor_to_pil_image(image_tensor)

        gt_label_idx = ground_truth_labels[idx_image]
        pred_label_idx = predicted_labels[idx_image]

        gt_label = get_key_by_value(label_mapping, int(gt_label_idx))
        pred_label = get_key_by_value(label_mapping, pred_label_idx)

        ax = fig.add_subplot(4, 4, idx_image + 1)
        plt.imshow(image_pil)
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

        text = f'GT is {gt_label}\nPred is {pred_label}'
        color = 'green' if gt_label == pred_label else 'red'
        ax.set_xlabel(text, color=color)

    plt.show()


def main():

    model = Model().to(device)

    batch_size = 24  # Set an appropriate batch size
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # Load the trained model
    checkpoint = torch.load('models/checkpoint.pkl', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    precision, recall, f1_score, TP, TN, FP, FN = evaluate_model(model, test_loader, device)

    print(f'TP = {TP}, TN = {TN}, FP = {FP}, FN = {FN}')
    print(f'Precision = {precision}, Recall = {recall}, F1 score = {f1_score}')

    # Display images with labels
    inputs, labels_gt = next(iter(test_loader))
    inputs = inputs[:16].to(device)  # Display the first 16 images
    labels_gt = labels_gt[:16].numpy()

    # Get predicted labels
    labels_predicted = model(inputs)
    predicted_probabilities = F.softmax(labels_predicted, dim=1).tolist()
    predicted_indexes = [torch.argmax(torch.tensor(probabilities)).item() for probabilities in predicted_probabilities]
    predicted_labels = [index for index in predicted_indexes]

    labels_gt = [str(label) for label in labels_gt]


    display_images_with_labels(inputs.cpu(), labels_gt, predicted_labels)

if __name__ == "__main__":
    main()
