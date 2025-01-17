"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import torch
import numpy as np

from utils.config import create_config
from utils.common_config import get_criterion, get_model, get_train_dataset, \
    get_val_dataset, get_train_dataloader, \
    get_val_dataloader, get_train_transformations, \
    get_val_transformations, get_optimizer, \
    adjust_learning_rate
from utils.evaluate_utils import contrastive_evaluate
from utils.memory import MemoryBank
from utils.train_utils import simclr_train
from utils.utils import fill_memory_bank
from termcolor import colored


def main(
    repo_root: str,
    config_env: str = None,
    config_exp: str = None,
    seed=1,
    output_dir: str = None,
    data_dir: str = None,
    download: bool = False,
    pretrain_epochs: int = 100,
):
    """
    SimCLR training script (i.e. typiclust's representation learning).

    Args:
        config_env (str): Config file for the environment.
        config_exp (str): Config file for the experiment.
        seed (int): Random seed
        output_dir (str): Directory to save the results.
        data_dir (str): Directory where the data is stored.
    """

    # Retrieve config file
    p = create_config(config_env, config_exp, seed,
                      repo_dir=repo_root+'/repos/TypiClust2/scan/', output_dir=output_dir)
    p['epochs'] = pretrain_epochs
    print(colored(p, 'red'))

    # Model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    print('Model is {}'.format(model.__class__.__name__))
    print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print(model)
    model = model.cuda()

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue'))
    torch.backends.cudnn.benchmark = True

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    train_transforms = get_train_transformations(p)
    print('Train transforms:', train_transforms)
    val_transforms = get_val_transformations(p)
    print('Validation transforms:', val_transforms)
    train_dataset = get_train_dataset(p, train_transforms, to_augmented_dataset=True,
                                      split='train+unlabeled', root=data_dir, download=download)  #
    # Split
    # is for stl-10
    val_dataset = get_val_dataset(p, val_transforms, root=data_dir, download=download)
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(val_dataset)))

    # Memory Bank
    print(colored('Build MemoryBank', 'blue'))
    base_dataset = get_train_dataset(
        p, val_transforms,
        split='train', root=data_dir, download=download
    )  # Dataset w/o
    # augs for knn eval
    base_dataloader = get_val_dataloader(p, base_dataset)
    memory_bank_base = MemoryBank(len(base_dataset),
                                  p['model_kwargs']['features_dim'],
                                  p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_base.cuda()
    memory_bank_val = MemoryBank(len(val_dataset),
                                 p['model_kwargs']['features_dim'],
                                 p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_val.cuda()

    # Criterion
    print(colored('Retrieve criterion', 'blue'))
    criterion = get_criterion(p)
    print('Criterion is {}'.format(criterion.__class__.__name__))
    criterion = criterion.cuda()

    # Optimizer and scheduler
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    print(optimizer)

    # Checkpoint
    if os.path.exists(p['pretext_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['pretext_checkpoint']), 'blue'))
        checkpoint = torch.load(p['pretext_checkpoint'], map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        model.cuda()
        start_epoch = checkpoint['epoch']

    else:
        print(colored('No checkpoint file at {}'.format(p['pretext_checkpoint']), 'blue'))
        start_epoch = 0
        model = model.cuda()

    # Training
    print(colored('Starting main loop', 'blue'))
    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' % (epoch, p['epochs']), 'yellow'))
        print(colored('-' * 15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train
        print('Train ...')
        simclr_train(train_dataloader, model, criterion, optimizer, epoch)

        # Fill memory bank
        print('Fill memory bank for kNN...')
        fill_memory_bank(base_dataloader, model, memory_bank_base)

        # Evaluate (To monitor progress - Not for validation)
        print('Evaluate ...')
        top1 = contrastive_evaluate(val_dataloader, model, memory_bank_base)
        print('Result of kNN evaluation is %.2f' % (top1))

        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                    'epoch': epoch + 1}, p['pretext_checkpoint'])

        topk = 20
        print('Mine the nearest neighbors (Top-%d)' % (topk))
        indices, acc = memory_bank_base.mine_nearest_neighbors(topk)
        np.save(p['topk_neighbors_train_path'], indices)
        np.save(p['pretext_features'], memory_bank_base.pre_lasts.cpu().numpy())
        np.save(p['pretext_features'].replace('features', 'test_features'),
                memory_bank_val.pre_lasts.cpu().numpy())

    # Save final model
    torch.save(model.state_dict(), p['pretext_model'])

    # Mine the topk nearest neighbors at the very end (Train)
    # These will be served as input to the SCAN loss.
    print(colored('Fill memory bank for mining the nearest neighbors (train) ...', 'blue'))
    fill_memory_bank(base_dataloader, model, memory_bank_base)
    topk = 20
    print('Mine the nearest neighbors (Top-%d)' % (topk))
    indices, acc = memory_bank_base.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on train set is %.2f' % (topk, 100 * acc))
    np.save(p['topk_neighbors_train_path'], indices)
    # save features
    np.save(p['pretext_features'], memory_bank_base.pre_lasts.cpu().numpy())
    np.save(p['pretext_features'].replace('features', 'test_features'),
            memory_bank_val.pre_lasts.cpu().numpy())

    # Mine the topk nearest neighbors at the very end (Val)
    # These will be used for validation.
    print(colored('Fill memory bank for mining the nearest neighbors (val) ...', 'blue'))
    fill_memory_bank(val_dataloader, model, memory_bank_val)
    topk = 5
    print('Mine the nearest neighbors (Top-%d)' % (topk))
    indices, acc = memory_bank_val.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on val set is %.2f' % (topk, 100 * acc))
    np.save(p['topk_neighbors_val_path'], indices)


if __name__ == '__main__':
    import fire

    fire.Fire(main)
