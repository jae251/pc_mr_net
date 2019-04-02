import torch
import torch.nn as nn
import os
import numpy as np
from time import time

from net_def import PointCloudMapRegressionNet
from data.hdf_dataset_loader import HdfDataset
from utilities.metrics import EvaluationMetrics
from utilities.utils import on_colab


def custom_collate_fn(sample):
    ''' This function is only necessary for a batch size greater than 1.
    Since the data samples are of varying sizes and the pytorch data loader class cannot handle this,
    the samples need to be padded to the same size. This enables the use of the inbuilt parallel data loading.
    '''
    shapes = torch.Tensor([s[0].shape for s in sample])
    max_shapes = shapes[:, [2, 3]].max(0)[0]
    input, label = [], []
    for s in sample:
        shape = s[0].shape
        height_padding = int(max_shapes[0] - shape[2])
        width_padding = int(max_shapes[1] - shape[3])
        pad = nn.ZeroPad2d((0, width_padding, 0, height_padding))
        input.append(pad(s[0]))
        label.append(pad(s[1]))
    input = torch.cat(input, 0)
    label = torch.cat(label, 0)
    return input, label


BATCHSIZE = 2
NUM_WORKERS = 3

if __name__ == '__main__':
    t1 = time()
    loss_function = nn.MSELoss()
    if on_colab():
        WEIGHT_DIR = "/content/drive/My Drive/net_weights"
        EVAL_DIR = "/content/drive/My Drive/eval_summaries"
    else:
        WEIGHT_DIR = "net_weights"
        EVAL_DIR = "eval_summaries"
    tensorboard_metrics = EvaluationMetrics(log_dir=EVAL_DIR)

    print("")
    for n, weight_file in enumerate(sorted(os.listdir(WEIGHT_DIR))):
        print("\rEvaluating ", weight_file)
        net = PointCloudMapRegressionNet()
        net.load_state_dict(torch.load(os.path.join(WEIGHT_DIR, weight_file)))
        net = net.cuda()

        eval_dataset = HdfDataset("../../data/dataset_one_car/eval")
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=BATCHSIZE, shuffle=False,
                                                  collate_fn=custom_collate_fn, num_workers=NUM_WORKERS)

        for i, (input, labels) in enumerate(eval_loader):
            if i % 10 == 0:
                print("\r{}".format(i), end="")
            input = input.cuda()
            labels = labels.cuda()

            output = net(input).data
            mask = labels != 0
            mask = mask.any(1)
            average_error_distance = ((output - labels) ** 2).sum(1).sqrt()
            average_error_distance = average_error_distance[mask]
            loss = loss_function(output, labels).item()
            tensorboard_metrics.add_batch(loss, output, labels)
        tensorboard_metrics.create_summary(n)
    t2 = time()
    print("\rElapsed evalutation time: {}".format(t2 - t1))
    tensorboard_metrics.close()
