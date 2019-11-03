import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


def load_pretrained(model, default_cfg, num_classes=1000, in_chans=3, filter_fn=None):
    if 'url' not in default_cfg or not default_cfg['url']:
        logging.warning("Pretrained model URL is invalid, using random initialization.")
        return

    state_dict = model_zoo.load_url(default_cfg['url'])

    if in_chans == 1:
        conv1_name = default_cfg['first_conv']
        logging.info('Converting first conv (%s) from 3 to 1 channel' % conv1_name)
        conv1_weight = state_dict[conv1_name + '.weight']
        state_dict[conv1_name + '.weight'] = conv1_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        assert False, "Invalid in_chans for pretrained weights"

    strict = True
    classifier_name = default_cfg['classifier']
    if num_classes == 1000 and default_cfg['num_classes'] == 1001:
        # special case for imagenet trained models with extra background class in pretrained weights
        classifier_weight = state_dict[classifier_name + '.weight']
        state_dict[classifier_name + '.weight'] = classifier_weight[1:]
        classifier_bias = state_dict[classifier_name + '.bias']
        state_dict[classifier_name + '.bias'] = classifier_bias[1:]
    elif num_classes != default_cfg['num_classes']:
        # completely discard fully connected for all other differences between pretrained and created model
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']
        strict = False

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    model.load_state_dict(state_dict, strict=strict)


def swish(x, inplace=False):
    if inplace:
        return x.mul_(x.sigmoid())
    else:
        return x * x.sigmoid()


def sigmoid(x, inplace=False):
    return x.sigmoid_() if inplace else x.sigmoid()


def drop_connect(inputs, training=False, drop_connect_rate=0.):
    """Apply drop connect."""
    if not training:
        return inputs

    keep_prob = 1 - drop_connect_rate
    random_tensor = keep_prob + torch.rand(
        (inputs.size()[0], 1, 1, 1), dtype=inputs.dtype, device=inputs.device)
    random_tensor.floor_()  # binarize
    output = inputs.div(keep_prob) * random_tensor
    return output


def load_checkpoint(model, checkpoint_path, strict=True):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        state_dict_key = ''
        if isinstance(checkpoint, dict):
            state_dict_key = 'model_state_dict'
        if state_dict_key and state_dict_key in checkpoint:
            model.load_state_dict(checkpoint[state_dict_key], strict=strict)
        else:
            model.load_state_dict(checkpoint, strict=strict)
        logging.info("Loaded {} from checkpoint '{}'".format(state_dict_key or 'weights', checkpoint_path))
    else:
        logging.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def seed_fn(worker_id=0):
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def distillate_data(loader, model, default_transform):
    dataset = loader.dataset
    dataset.transform = default_transform

    outputs = []
    losses = []
    idxs = []
    targets = []
    criterion = nn.CrossEntropyLoss(reduction='none')

    for batch in tqdm(loader):
        for key in batch:
            batch[key] = batch[key].cuda()

        with torch.no_grad():
            output = model.forward(batch['input'])
            loss = criterion(output, batch['target'])
        outputs.append(output.cpu())
        losses.append(loss.cpu())
        targets.append(batch['target'].cpu())
        idxs.append(batch['index'].cpu())

    outputs = nn.functional.softmax(torch.cat(outputs), dim=1)
    losses = torch.cat(losses)
    idxs = torch.cat(idxs)
    targets = torch.cat(targets)

    score_idxs = torch.argsort(losses)
    outputs = outputs[score_idxs]
    idxs = idxs[score_idxs]

    mx, argmx = outputs.max(dim=1)
    label = loader.dataset.label
    criter = (argmx != torch.from_numpy(label)) & (mx > 0.8)

    idxs_wrong = idxs[criter]
    label[idxs_wrong.numpy()] = argmx[criter].numpy()

    idxs_left = idxs[outputs.max(dim=1)[0] > 0.5]
    loader.dataset.data = loader.dataset.data[idxs_left.numpy()]
    loader.dataset.label = label[idxs_left.numpy()]


class GeneralizedCrossEntropyLoss(nn.Module):
    def __init__(self, q=0.7):
        super(GeneralizedCrossEntropyLoss, self).__init__()
        self.q = q

    def forward(self, input, target):
        """
        This loss function is proposed in:
         Zhilu Zhang and Mert R. Sabuncu, "Generalized Cross Entropy Loss for Training Deep Neural Networks with
         Noisy Labels", 2018
        https://arxiv.org/pdf/1805.07836.pdf
        """
        probs = torch.softmax(input, dim=1)[:, target]
        return torch.mean((1 - (probs + 1e-8) ** self.q) / self.q)

    # criterion = GeneralizedCrossEntropyLoss()

    # criterions = {
    #     'source_train': nn.CrossEntropyLoss(),
    #     'source_valid': nn.CrossEntropyLoss(),
    #     'target_train': GeneralizedCrossEntropyLoss(),
    #     'target_valid': GeneralizedCrossEntropyLoss(),
    # }
    # criterions = {
    #     'source_train': nn.CrossEntropyLoss(),
    #     'source_valid': nn.CrossEntropyLoss(),
    #     'target_train': nn.CrossEntropyLoss(),
    #     'target_valid': nn.CrossEntropyLoss(),
    # }
