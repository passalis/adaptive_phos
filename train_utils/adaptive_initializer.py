import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
import torch.nn as nn
import torch
from tqdm import tqdm


def epoch_trainer_adaptive(model, loader, lr=0.01, optimizer=optim.RMSprop, scaler=None, projector=None, layer=0,
                           subsample=5):
    params = list([scaler])
    params.extend(list(projector.parameters()))
    model_optimizer = optimizer(params, lr=lr)

    model.train()

    criterion = CrossEntropyLoss()

    train_loss, counter = 0, 0

    for (inputs, targets) in loader:
        # Reset gradients
        model_optimizer.zero_grad()

        # Get the data
        inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())

        inputs = inputs[:, -subsample:, :]
        targets = torch.squeeze(targets)

        # Feed forward the network and update
        outputs = model(inputs, scaler=scaler, layer=layer)
        outputs = projector(outputs)
        loss = criterion(outputs, targets)

        loss.backward()
        model_optimizer.step()

        # Calculate statistics
        train_loss += loss.item()
        counter += inputs.size(0)

    loss = (loss / counter).cpu().data.numpy()
    return loss


def per_layer_calculation(model, loader, train_epochs=10, layer=0):
    projector = nn.Linear(model.projector_sizes[layer][0], model.projector_sizes[layer][1]).cuda()
    scaler = nn.Parameter(torch.FloatTensor([1.0]).cuda())
    for epoch in tqdm(range(train_epochs)):
        epoch_trainer_adaptive(model=model, loader=loader, layer=layer, scaler=scaler, projector=projector, lr=0.1)
    return scaler.item()


def adaptive_initialization(model, loader, train_epochs, n_layers=3, initializer=None):
    scaling_factors = []

    current_model = model()
    current_model.cuda()

    if initializer is not None:
        current_model.apply(initializer)

    for i in range(0, n_layers):
        cur_scale = per_layer_calculation(current_model, loader, train_epochs, layer=i)
        cur_std = torch.std(current_model.weights[i]).cpu().detach().item() * torch.abs(cur_scale)
        print("Initializing layer ", i, " with  = ", cur_std)
        current_model.weights[i].data.normal_(0, cur_std)
        scaling_factors.append(cur_scale)

    return scaling_factors, current_model
