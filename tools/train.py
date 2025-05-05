import torch
import argparse
import os
import numpy as np
import yaml
import random
from tqdm import tqdm
from model.detr import DETR
from dataset.voc import VOCDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')


def collate_function(data):
    return tuple(zip(*data))


def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    #########################

    dataset_config = config['dataset_params']
    train_config = config['train_params']
    model_config = config['model_params']

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    voc = VOCDataset('train',
                     im_sets=dataset_config['train_im_sets'],
                     im_size=dataset_config['im_size'])
    train_dataset = DataLoader(voc,
                               batch_size=train_config['batch_size'],
                               shuffle=True,
                               collate_fn=collate_function)

    # Instantiate model and load checkpoint if present
    model = DETR(
        config=model_config,
        num_classes=dataset_config['num_classes'],
        bg_class_idx=dataset_config['bg_class_idx']
    )
    model.to(device)
    model.train()

    if os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['ckpt_name'])):

        state_dict = torch.load(
            os.path.join(train_config['task_name'],
                         train_config['ckpt_name']),
            map_location=device)
        model.load_state_dict(state_dict)
        print('Loading checkpoint as one exists')

    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    optimizer = torch.optim.AdamW(lr=train_config['lr'],
                                  params=filter(lambda p: p.requires_grad,
                                                model.parameters()),
                                  weight_decay=1E-4)

    # backbone_params = [
    #     p for n, p in model.named_parameters() if 'backbone.' in n]
    # transformer_params = [
    #     p for n, p in model.named_parameters() if 'backbone.' not in n]
    # optimizer = torch.optim.AdamW([
    #     {'params': backbone_params, 'lr': train_config['lr']*0.1},
    #     {'params': transformer_params, 'lr': train_config['lr']},
    # ], weight_decay=1e-4)

    lr_scheduler = MultiStepLR(optimizer,
                               milestones=train_config['lr_steps'],
                               gamma=0.1)
    acc_steps = train_config['acc_steps']
    num_epochs = train_config['num_epochs']
    steps = 0
    for i in range(num_epochs):
        detr_classification_losses = []
        detr_localization_losses = []
        for idx, (ims, targets, _) in enumerate(tqdm(train_dataset)):
            for target in targets:
                target['boxes'] = target['boxes'].float().to(device)
                target['labels'] = target['labels'].long().to(device)
            images = torch.stack([im.float().to(device) for im in ims], dim=0)
            batch_losses = model(images, targets)['loss']

            loss = (sum(batch_losses['classification']) +
                    sum(batch_losses['bbox_regression']))

            detr_classification_losses.append(sum(batch_losses['classification']).item())
            detr_localization_losses.append(sum(batch_losses['bbox_regression']).item())
            loss = loss / acc_steps
            loss.backward()

            if (idx + 1) % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            if steps % train_config['log_steps'] == 0:
                loss_output = ''
                loss_output += 'DETR Classification Loss : {:.4f}'.format(
                    np.mean(detr_classification_losses))
                loss_output += ' | DETR Localization Loss : {:.4f}'.format(
                    np.mean(detr_localization_losses))
                print(loss_output, lr_scheduler.get_last_lr())
            if torch.isnan(loss):
                print('Loss is becoming nan. Exiting')
                exit(0)
            steps += 1
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
        print('Finished epoch {}'.format(i+1))
        loss_output = ''
        loss_output += 'DETR Classification Loss : {:.4f}'.format(
            np.mean(detr_classification_losses))
        loss_output += ' | DETR Localization Loss : {:.4f}'.format(
            np.mean(detr_localization_losses))
        print(loss_output)
        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                         train_config['ckpt_name']))
    print('Done Training...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for detr training')
    parser.add_argument('--config', dest='config_path',
                        default='config/voc.yaml', type=str)
    args = parser.parse_args()
    train(args)
