import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from models.effnet.effnet import efficientnet_b0
from datasets import prepare_data_loaders
from config import config
from models.utils import seed_fn


def train_cycle(num_epoch, loaders, model, optimizer, criterion, scheduler, metrics, best_result):
    for epoch in range(num_epoch):
        print(f'Start epoch {epoch}')
        for mode, loader in loaders.items():
            train_mode = mode.endswith('train')
            if train_mode:
                model.train()
            else:
                model.eval()
            total_batches = len(loader)
            t = tqdm(enumerate(loader), total=total_batches)
            for i, batch in t:
                optimizer.zero_grad()

                with torch.set_grad_enabled(train_mode):
                    if 'index' in batch:
                        indices = batch.pop('index')
                    for key in batch:
                        batch[key] = batch[key].cuda()

                    if train_mode:
                        output = model.forward(**batch)
                    else:
                        output = model.forward(batch['input'])

                    loss = criterion(output, batch['target'])
                    loss_value = float(loss.data)

                    if train_mode:
                        loss.backward()
                        optimizer.step()
                        if isinstance(scheduler, lr_scheduler.CyclicLR):
                            scheduler.step()

                    metrics[-1].update(loss_value)
                    for metric in metrics[:-1]:
                        metric.update(output, batch['target'])

                t.set_description_str(desc=f'Loss={metrics[-1].avg:.3f}', refresh=False)
            t.close()
            print(f'{mode} loss: {metrics[-1].avg:.3f}')
            for metric in metrics[:-1]:
                print(f'{metric.name()}: {metric.value()}')

            if mode == 'target_valid':
                new_result = metrics[0].value()
                print(best_result, new_result)
                if best_result <= new_result:
                    best_result = new_result
                    torch.save(model.state_dict(), f'checkpoints/effnet_{int(round(best_result*1000))}.pth')
                if not (scheduler is None or isinstance(scheduler, lr_scheduler.CyclicLR)):
                    scheduler.step()

            for metric in metrics:
                metric.reset()
        print()


seed_fn()

learning_rate = 0.01
num_epoch_first_train = 11
num_epoch_fine_tune = 4
pretrained = True

mnist_loader = prepare_data_loaders(config['mnist'], ['valid'])
svhn_loaders = prepare_data_loaders(config['svhn'], ['train', 'valid'])
loaders = {
    'source_train': svhn_loaders['train'],
    'source_valid': svhn_loaders['valid'],
    'target_valid': mnist_loader['valid'],
}

model = efficientnet_b0(pretrained=pretrained, num_classes=10).cuda()

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 5, 9], gamma=0.1)

metrics = config['metrics']

num_batch = len(svhn_loaders['train'])
criterion = nn.CrossEntropyLoss()

best_result = 0

default_transform = config['svhn']['valid']['transforms']
train_cycle(num_epoch_first_train, loaders, model, optimizer, criterion, scheduler, metrics, best_result)
