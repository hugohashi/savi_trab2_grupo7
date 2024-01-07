#!/usr/bin/env python3

import json
import torch
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt

from dataset import Dataset
from model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('json_files/dataset_files.json', 'r') as f:
    dataset_filenames = json.load(f)

test_filenames = dataset_filenames['testing_files']

test_dataset = Dataset(test_filenames)

print(f'Used {len(test_filenames)} for testing ')

with open('json_files/label_mapping.json', 'r') as f:
    label_mapping = json.load(f)


# Get the key of a dictionary from its value 
def get_key_by_value(dictionary, target_value):
    for key, value in dictionary.items():
        if value == target_value:
            return key
    # If the value is not found, return None
    return None

# Calculate metrics for the model (True Positive, True Negative, False Positive and False Negative)
def calculate_metrics_per_class(confusion_mat):
    num_classes = confusion_mat.shape[0]
    metrics_per_class = []

    for i in range(num_classes):
        TP = confusion_mat[i, i]
        FP = sum(confusion_mat[:, i]) - TP
        FN = sum(confusion_mat[i, :]) - TP
        TN = sum(sum(confusion_mat)) - TP - FP - FN

        precision = TP / (TP + FP) if TP + FP != 0 else 0
        recall = TP / (TP + FN) if TP + FN != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

        metrics_per_class.append({
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN
        })

    return metrics_per_class


def evaluate_model(model, test_loader, device, num_classes):
    model.eval()

    all_predicted_labels = []
    all_ground_truth_labels = []

    results_list = []

    with torch.no_grad():
        for inputs, labels_gt, filenames in test_loader:
            inputs = inputs.to(device)
            labels_gt = labels_gt.clone().detach().to(device)

            labels_predicted = model(inputs)
            predicted_probabilities = F.softmax(labels_predicted, dim=1)
            predicted_indexes = torch.argmax(predicted_probabilities, dim=1).cpu().numpy()

            for filename, gt_label, pred_label in zip(filenames, labels_gt.cpu().numpy(), predicted_indexes):
                result_dict = {
                    'filename': filename,
                    'ground_truth': get_key_by_value(label_mapping, gt_label),
                    'predicted': get_key_by_value(label_mapping, pred_label)
                }
                results_list.append(result_dict)

            all_predicted_labels.extend(predicted_indexes)
            all_ground_truth_labels.extend(labels_gt.cpu().numpy())

    confusion_mat = confusion_matrix(all_ground_truth_labels, all_predicted_labels, labels=range(num_classes))
    metrics_per_class = calculate_metrics_per_class(confusion_mat)

    return metrics_per_class, results_list


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

    batch_size = 24
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # Load the trained model
    checkpoint = torch.load('models/model.pkl', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    metrics_per_class, results_list = evaluate_model(model, test_loader, device, num_classes=len(label_mapping))
    
    # Convert torch.int64 to Python int for JSON serialization
    metrics_per_class_json = []
    for i, metrics in enumerate(metrics_per_class):
        class_metrics = {
            'class_index': i,
            'class_label': get_key_by_value(label_mapping, i),
            'metrics': {
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1_score': float(metrics['f1_score']),
                'TP': int(metrics['TP']),
                'TN': int(metrics['TN']),
                'FP': int(metrics['FP']),
                'FN': int(metrics['FN'])
            }
        }
        metrics_per_class_json.append(class_metrics)

    # Save metrics to a JSON file
    metrics_filename = 'json_files/metrics.json'
    with open(metrics_filename, 'w') as metrics_file:
        json.dump(metrics_per_class_json, metrics_file, indent=2)
    print(f'Metrics saved to {metrics_filename}')

    for i, metrics in enumerate(metrics_per_class):
        print(f'\nMetrics for class {i}:')
        print(f'TP = {metrics["TP"]}, TN = {metrics["TN"]}, FP = {metrics["FP"]}, FN = {metrics["FN"]}')
        print(f'Precision = {metrics["precision"]}, Recall = {metrics["recall"]}, F1 score = {metrics["f1_score"]}')

    results_filename = 'json_files/results.json'
    with open(results_filename, 'w') as results_file:
        json.dump(results_list, results_file, indent=2)
    print(f'Results saved to {results_filename}')

    # Display images with labels
    inputs, labels_gt, _ = next(iter(test_loader))
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
