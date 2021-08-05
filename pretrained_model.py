import os

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import time
import shutil

from DataLoaders import load_gen_data_dir
from ModelTraining import eval_model, dump_to_log


DATASETS_DIR = "Datasets"


def main_alex_net(dataset_name, unlabeled_train, unlabeled_val, labeled_train, labeled_val,
                  classes, fold_num, out_dir, epochs=40):
    AlexNet_Model = torchvision.models.alexnet(pretrained=True)   # torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    AlexNet_Model.eval()
    AlexNet_Model.classifier[1] = nn.Linear(9216, 4096)
    AlexNet_Model.classifier[4] = nn.Linear(4096, 1024)
    AlexNet_Model.classifier[6] = nn.Linear(1024, len(classes))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    AlexNet_Model.to(device)
    log_name = f"{out_dir}/AlexNet_{dataset_name}_log.log"
    train_time = train_alex_net(AlexNet_Model, labeled_train, epochs, device)
    dump_to_log({fold_num: {"train_time": train_time}}, log_name)
    eval_ans = eval_model(AlexNet_Model, "AlexNet_Model", [unlabeled_val, labeled_val], classes, fold_num,
                          dataset_name, out_dir=out_dir)
    dump_to_log(eval_ans, log_name)


def train_alex_net(AlexNet_Model, train_loader, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(AlexNet_Model.parameters(), lr=0.001, momentum=0.9)
    start_time = time.time()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            output = AlexNet_Model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    end_time = time.time()
    print('Finished Training of AlexNet')
    return end_time - start_time


# def eval_alex_net(AlexNet_Model, test_loader, device, classes, fold_num, dat):
    # AlexNet_Model.eval()
    # # Testing Accuracy
    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for data in test_loader:
    #         images, labels = data[0].to(device), data[1].to(device)
    #         outputs = AlexNet_Model(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #
    # print('Accuracy of the network on the 10000 test images: %.2f %%' % (100 * correct / total))
    #
    # class_correct = list(0. for i in range(10))
    # class_total = list(0. for i in range(10))
    # with torch.no_grad():
    #     for data in test_loader:
    #         images, labels = data[0].to(device), data[1].to(device)
    #         outputs = AlexNet_Model(images)
    #         _, predicted = torch.max(outputs, 1)
    #         c = (predicted == labels).squeeze()
    #         for i in range(4):
    #             label = labels[i]
    #             class_correct[label] += c[i].item()
    #             class_total[label] += 1
    #
    # for i in range(10):
    #     print('Accuracy of %5s : %2d %%' % (
    #         classes[i], 100 * class_correct[i] / class_total[i]))
    #
    # # Verifying average accuracy of the network
    # avg = 0
    # for i in range(10):
    #     temp = (100 * class_correct[i] / class_total[i])
    #     avg = avg + temp
    # avg = avg / 10
    # print('Average accuracy = ', avg)


def main():
    out_dir = "outputs_alex"
    try:
        shutil.rmtree(out_dir, ignore_errors=True)
    except FileNotFoundError:
        pass
    os.makedirs(out_dir, exist_ok=True)

    for dataset_name in os.listdir(DATASETS_DIR):
        k_fold = 10
        unlabeled_gen, labeled_gen, classes = load_gen_data_dir(os.path.join(DATASETS_DIR, dataset_name), k_fold,
                                                                resize=(256, 265))

        for fold_num, ((unlabeled_train, unlabeled_val), (labeled_train, labeled_val)) in enumerate(
                zip(unlabeled_gen, labeled_gen)):
            main_alex_net(dataset_name, unlabeled_train, unlabeled_val, labeled_train, labeled_val,
                          classes, fold_num, out_dir=out_dir)


if __name__ == '__main__':
    main()
