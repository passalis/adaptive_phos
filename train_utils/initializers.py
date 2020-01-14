import torch
from torch.nn.init import xavier_normal_, kaiming_normal_


def create_custom_initializer(std):
    return lambda m: weights_init_normal(m, std)


def weights_init_normal(m, std=0.5):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, std)
        m.bias.data.fill_(0)


def weights_xavier_normal(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        xavier_normal_(m.weight)
        m.bias.data.fill_(0)


def weights_kaiming_normal(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        kaiming_normal_(m.weight)
        m.bias.data.fill_(0)
