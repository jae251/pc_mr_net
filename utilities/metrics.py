import os
from tensorboardX import SummaryWriter


class Metrics:
    def __init__(self, log_dir=None):  # , metrics=[]):
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)

    def create_summary(self, loss, output, labels, n):
        # Metric: training loss
        self.writer.add_scalar("Training Loss", loss, n)
        # Metric: average miss distance and accuracy
        mask = labels != 0
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


if __name__ == '__main__':
    metrics = Metrics()
