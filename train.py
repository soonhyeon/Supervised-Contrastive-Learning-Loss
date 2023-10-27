import torch 
import torch.optim as optim
from tqdm import tqdm
from resnet import SupConResNet, model_dict
from supcontrast import SupConLoss
from dataloader import cifar10_loader


def train(train_loader, model, criterion, optimizer, epoch, method='SupCon'):
  num_steps = len(train_loader)

  for _ in range(epoch):
    total_loss = 0.

    for _, (images, labels) in tqdm(enumerate(train_loader), total=num_steps):
      # images[0].shape: [B, 3, 32, 32], images[1].shape: [B, 3, 32, 32]
      labels = labels.to('cuda')

      images = torch.cat([images[0], images[1]], dim=0).to('cuda') # images.shape: [2B, 3, 32, 32]

      bsz = labels.shape[0] # bsz: B

      features = model(images) # [2B, 128]

      f1, f2 = torch.split(features, [bsz, bsz], dim=0) # f1: [B, 128], f2: [B, 128]

      features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1) # [B, 2, 128]

      if method == 'SupCon':
        loss = criterion(features, labels)

      elif method == 'SimCLR':
        loss = criterion(features)

      else:
        raise ValueError(f'contrastive method not supported: {method}')

      total_loss += loss.item()

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    print(f'Epoch {_+1} loss value: {total_loss / len(train_loader)}')


def main():
    model = SupConResNet(model_dim_dict=model_dict['resnet50'])
    criterion = SupConLoss(temperature=0.07, contrast_mode='all', base_temperature=0.07)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
    train_loader = cifar10_loader()
    
    train(train_loader=train_loader, model=model, 
          criterion=criterion, optimizer=optimizer, epoch=100)


if __name__ == '__main__':
    main()