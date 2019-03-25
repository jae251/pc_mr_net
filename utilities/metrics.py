import os
import numpy as np
import torch
from tensorboardX import SummaryWriter


class TrainingMetrics:
    def __init__(self, log_dir=None):
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)

    def create_summary(self, loss, output, labels, n):
        # Metric: training loss
        self.writer.add_scalar("Training Loss", loss, n)
        # Metric: average miss distance and accuracy
        mask = labels != 0  # only look at slots with data, does not account for predictions for empty slots
        mask = mask.any(1)
        error_distance = ((output - labels) ** 2).sum(1).sqrt()
        error_distance = error_distance[mask]
        inv_samples = 1 / len(error_distance)
        average_error_distance = float(error_distance.sum()) * inv_samples
        self.writer.add_scalar("Average Miss Distance", average_error_distance, n)
        self.writer.add_scalar("Accuracy inside 2m Diameter", float((error_distance < 1).sum()) * inv_samples, n)
        self.writer.add_scalar("Accuracy inside 1m Diameter", float((error_distance < .5).sum()) * inv_samples, n)
        self.writer.add_scalar("Accuracy inside 0.1m Diameter", float((error_distance < .05).sum()) * inv_samples, n)
        self.writer.add_histogram("Error Distance Histogram", error_distance, n)

    def close(self):
        self.writer.close()


class EvaluationMetrics:
    '''
    In contrast to TrainingMetrics, this metric writer accumulates metrics with add_batch
    and records the average to a summary file only when create_summary is called
    '''

    def __init__(self, log_dir=None):
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.batches = []  # holds the metrics for each batch, as well as the batch size
        self.error_distances = []

    def add_batch(self, loss, output, labels):
        loss_b = (loss, 1)
        mask = labels != 0
        mask = mask.any(1)
        error_distance = ((output - labels) ** 2).sum(1).sqrt()
        error_distance = error_distance[mask]
        len_samples = len(error_distance)
        error_distance_b = (float(error_distance.sum()), len_samples)
        acc_2m_b = (float((error_distance < 1).sum()), len_samples)
        acc_1m_b = (float((error_distance < .5).sum()), len_samples)
        acc_01m_b = (float((error_distance < .05).sum()), len_samples)
        self.batches.append([loss_b, error_distance_b, acc_2m_b, acc_1m_b, acc_01m_b])
        self.error_distances.append(error_distance)

    def create_summary(self, n):
        metric_batches = np.array(self.batches)
        reduced_metrics = (metric_batches[:, :, 0] * metric_batches[:, :, 1]).sum(axis=0) \
                          / metric_batches[:, :, 1].sum(axis=0)
        error_distances = torch.cat(self.error_distances)
        self.writer.add_scalar("Training Loss", reduced_metrics[0], n)
        self.writer.add_scalar("Average Miss Distance", reduced_metrics[1], n)
        self.writer.add_scalar("Accuracy inside 2m Diameter", reduced_metrics[2], n)
        self.writer.add_scalar("Accuracy inside 1m Diameter", reduced_metrics[3], n)
        self.writer.add_scalar("Accuracy inside 0.1m Diameter", reduced_metrics[4], n)
        self.writer.add_histogram("Error Distance Histogram", error_distances, n)
        self.batches = []
        self.error_distances = []

    def close(self):
        self.writer.close()
