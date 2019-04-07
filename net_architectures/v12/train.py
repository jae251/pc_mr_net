from time import time
import os
import shutil
import torch
import torch.nn as nn
from torch.optim import Adam

from net_def import PointCloudMapRegressionNet
from data.hdf_dataset_loader import HdfDataset
from utilities.metrics import TrainingMetrics
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


EPOCHS = 1000
BATCHSIZE = 10

if __name__ == '__main__':
    t1 = time()
    net = PointCloudMapRegressionNet()

    on_colab = on_colab()
    if on_colab:
        MODEL_DIR = "/content/drive/My Drive/net_weights"
        from multiprocessing import cpu_count

        NUM_WORKERS = cpu_count()
    else:
        MODEL_DIR = "net_weights"
        NUM_WORKERS = 3
    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    SUMMARY_DIR = "summaries"
    if not os.path.isdir(SUMMARY_DIR):
        os.mkdir(SUMMARY_DIR)

    nr_saved_models = len(os.listdir(MODEL_DIR))
    LOAD_MODEL = os.path.join(MODEL_DIR, "{:04}.pt".format(nr_saved_models))
    nr_saved_models += 1
    SAVE_MODEL = os.path.join(MODEL_DIR, "{:04}.pt".format(nr_saved_models))

    if os.path.isfile(LOAD_MODEL):
        print("Loaded parameters from ", LOAD_MODEL)
        net.load_state_dict(torch.load(LOAD_MODEL))
        ep_done = (nr_saved_models - 1) * 10  # -1 due to saving of initial parameter conditions
    else:
        ep_done = 0
    net = net.cuda()
    optimizer = Adam(net.parameters())
    loss_function = nn.MSELoss()
    tensorboard_metrics = TrainingMetrics(log_dir="summaries")
    # tensorboard_metrics = Metrics(log_dir="/content/drive/My Drive/summaries")

    train_dataset = HdfDataset("../../data/dataset_one_car/train")
    eval_dataset = HdfDataset("../../data/dataset_one_car/eval")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True,
                                               collate_fn=custom_collate_fn, num_workers=NUM_WORKERS)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=BATCHSIZE, shuffle=False,
                                              collate_fn=custom_collate_fn, num_workers=NUM_WORKERS)

    nr_training_samples = len(train_loader)
    print('==>>> total training batch number: {}'.format(nr_training_samples))
    print('==>>> total testing batch number: {}'.format(len(eval_loader)))

    # training loop
    metric_periodicity = 10
    for epoch in range(EPOCHS):
        print("\rEPOCH ", epoch)
        for i, (input, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            input = input.cuda()
            labels = labels.cuda()

            output = net(input)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()
            if i % metric_periodicity == 0:
                print("\r{}".format(i), end="")
                iteration = (nr_training_samples * (epoch + ep_done) + i) / metric_periodicity
                tensorboard_metrics.create_summary(loss, output, labels, iteration)
        # if epoch % 10 == 0:
        torch.save(net.state_dict(), SAVE_MODEL)
        print("\rSaved model in ", SAVE_MODEL)
        nr_saved_models += 1
        SAVE_MODEL = os.path.join(MODEL_DIR, "{:04}.pt".format(nr_saved_models))
        if on_colab:
            tensorboard_metrics.close()
            summary_file = os.listdir("summaries")[0]
            shutil.move(os.path.join("summaries", summary_file),
                        os.path.join("/content/drive/My Drive/summaries", summary_file))
            tensorboard_metrics = TrainingMetrics(log_dir="summaries")
    t2 = time()
    print("Elapsed training time: {}".format(t2 - t1))

    # evaluation loop
    loss = 0
    samples = 0
    for i, (input, labels) in enumerate(eval_loader):
        if i % 10 == 0:
            print("\r{}".format(i), end="")
        input = input.cuda()
        labels = labels.cuda()

        output = net(input)
        mask = labels != 0
        mask = mask.any(1)
        average_error_distance = ((output - labels) ** 2).sum(1).sqrt()
        average_error_distance = average_error_distance[mask]
        samples += len(average_error_distance)
        loss += float(average_error_distance.sum())
    t3 = time()
    print("")
    print("Elapsed evalutation time: {}".format(t3 - t2))
    print("Average loss is: ", loss / samples)

    torch.save(net.state_dict(), SAVE_MODEL)
    print("Saved model in ", SAVE_MODEL)
