from lob_utils.lob_loader import get_wf_lob_loaders
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score
import numpy as np
from train_utils.adaptive_initializer import adaptive_initialization
from torch.autograd import Variable
import torch
from tqdm import tqdm
from torch.autograd import Variable
import torch.optim as optim
from torch.nn import CrossEntropyLoss


def train_evaluate_anchored(model, horizon=0, window=5, batch_size=128, train_epochs=20, verbose=True,
                            learning_rate=0.0001, splits=[6, 7, 8], use_adaptive_init=False, weight_initializer=None,
                            adaptive_init_iters=10):
    results = []

    for i in splits:
        print("Evaluating for split: ", i)
        train_loader, test_loader = get_wf_lob_loaders(window=window, horizon=horizon, split=i, batch_size=batch_size,
                                                       class_resample=True, normalization='std', shift=None)

        if use_adaptive_init:
            scaling_factors, current_model = adaptive_initialization(model, train_loader, adaptive_init_iters,
                                                                     initializer=weight_initializer)
        else:
            current_model = model()
            current_model.cuda()
            if weight_initializer:
                current_model.apply(weight_initializer)

        for epoch in range(train_epochs):
            loss = epoch_trainer(model=current_model, loader=train_loader, lr=learning_rate)
            if verbose:
                print("Epoch ", epoch, "loss: ", loss)

        test_results = evaluator(current_model, test_loader)
        print(test_results)
        results.append(test_results)

    return results


def get_average_metrics(results):
    precision, recall, f1 = [], [], []
    kappa = []
    acc = []
    for x in results:
        acc.append(x['accuracy'])
        precision.append(x['precision_avg'])
        recall.append(x['recall_avg'])
        f1.append(x['f1_avg'])
        kappa.append(x['kappa'])

    return acc, precision, recall, f1, kappa


def epoch_trainer(model, loader, lr=0.0001, optimizer=optim.RMSprop):
    model.train()

    model_optimizer = optimizer(params=model.parameters(), lr=lr)

    criterion = CrossEntropyLoss()

    train_loss, counter = 0, 0

    for (inputs, targets) in loader:
        # Reset gradients
        model_optimizer.zero_grad()

        # Get the data
        inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
        targets = torch.squeeze(targets)

        # Feed forward the network and update
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        model_optimizer.step()

        # Calculate statistics
        train_loss += loss.item()
        counter += inputs.size(0)

    loss = (loss / counter).cpu().data.numpy()
    return loss


def evaluator(model, loader):
    model.eval()
    true_labels = []
    predicted_labels = []

    for (inputs, targets) in tqdm(loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        predicted_labels.append(predicted.cpu().numpy())
        true_labels.append(targets.cpu().data.numpy())

    true_labels = np.squeeze(np.concatenate(true_labels))
    predicted_labels = np.squeeze(np.concatenate(predicted_labels))

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average=None)
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(true_labels, predicted_labels,
                                                                           average='macro')
    kappa = cohen_kappa_score(true_labels, predicted_labels)

    metrics = {}
    metrics['accuracy'] = np.sum(true_labels == predicted_labels) / len(true_labels)

    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1

    metrics['precision_avg'] = precision_avg
    metrics['recall_avg'] = recall_avg
    metrics['f1_avg'] = f1_avg

    metrics['kappa'] = kappa

    return metrics
