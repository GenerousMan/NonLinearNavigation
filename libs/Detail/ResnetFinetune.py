import os
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import time
from torchvision import models
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
torch.cuda.set_device(0)


def load_data(root_path, batch_size):
    transform_dict = {
        'src': transforms.Compose(
        [transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ]),
        'tar': transforms.Compose(
        [transforms.Resize(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ])}
    data = datasets.ImageFolder(root=root_path, transform=transform_dict['src'])
    print(data.class_to_idx)
    
    target = data.targets
    class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False,sampler = sampler)
    return data_loader


class TransferModel(nn.Module):
    def __init__(self,
                base_model : str = 'resnet50',
                pretrain : bool = True,
                n_class : int = 4):
        super(TransferModel, self).__init__()
        self.base_model = base_model
        self.pretrain = pretrain
        self.n_class = n_class
        if self.base_model == 'resnet50':
            self.model = torchvision.models.resnet50(pretrained=True)
            n_features = self.model.fc.in_features
            fc = torch.nn.Linear(n_features, n_class)
            self.model.fc = fc
        else:
            # Use other models you like, such as vgg or alexnet
            pass
        self.model.fc.weight.data.normal_(0, 0.005)
        self.model.fc.bias.data.fill_(0.1)

    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        return self.forward(x)



def finetune(model, dataloader, optimizer):
    since = time.time()
    best_acc = 0
    stop = 0
    for epoch in range(0, n_epoch):
        stop += 1
        # You can uncomment this line for scheduling learning rate
        # lr_schedule(optimizer, epoch)
        for phase in ['src','src']:
            if phase == 'src':
                model.train()
            else:
                model.eval()
            total_loss, correct = 0, 0
            batch_count = 0
            for inputs, labels in dataloader:
                # print('batch:',batch_count,'/',int(len(dataloader.dataset)/batch_size))
                # print('labels:',labels)
                inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'src'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                preds = torch.max(outputs, 1)[1]
                if phase == 'src':
                    loss.backward()
                    optimizer.step()
                # print('loss:',loss)
                # print('preds:',preds)
                # print('correct nums:',torch.sum(preds == labels.data))
                # print("_______________")
                total_loss += loss.item() * inputs.size(0)
                correct += torch.sum(preds == labels.data)
                batch_count+=1
            epoch_loss = total_loss / len(dataloader.dataset)
            epoch_acc = correct.double() / len(dataloader.dataset)
            print(f'Epoch: [{epoch:02d}/{n_epoch:02d}]---{phase}, loss: {epoch_loss:.6f}, acc: {epoch_acc:.4f}')
            if phase == 'src' and epoch_acc > best_acc:
                stop = 0
                best_acc = epoch_acc
                torch.save(model.state_dict(), str(n_class)+'_model.pkl')
        if stop >= early_stop:
            break
        print()
   
    time_pass = time.time() - since
    print(f'Training complete in {time_pass // 60:.0f}m {time_pass % 60:.0f}s')


if __name__ == "__main__":
    data_folder = '4_images'
    batch_size = 32
    n_class = 4

    dataloader = load_data(data_folder, batch_size)
    print('Source data number:', len(dataloader.dataset))
    model = TransferModel().cuda()
    RAND_TENSOR = torch.randn(1, 3, 224, 224).cuda()
    output = model(RAND_TENSOR)
    print(output)
    print(output.shape)
    n_epoch = 100
    criterion = nn.CrossEntropyLoss()
    early_stop = 20
    param_group = []
    learning_rate = 0.0001
    momentum = 5e-4
    for k, v in model.named_parameters():
        if not k.__contains__('fc'):
            param_group += [{'params': v, 'lr': learning_rate}]
        else:
            param_group += [{'params': v, 'lr': learning_rate * 10}]

    optimizer = torch.optim.SGD(param_group, momentum=momentum)


    finetune(model, dataloader, optimizer)