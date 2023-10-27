import torch 
import torch.nn as nn 


class SupConLoss(nn.Module):
  def __init__(self, temperature=0.07, contrast_mode = 'all', base_temperature=0.07):
    super(SupConLoss, self).__init__()
    self.temperature = temperature
    self.contrast_mode = contrast_mode
    self.base_temperature = base_temperature

  def forward(self, features, labels=None, mask=None):
    device = torch.device('cuda') if features.is_cuda else torch.device('cpu')
    batch_size = features.shape[0]
    if labels is None and mask is None:
      #SimCLR configuration
      mask = torch.eye(batch_size, dtype=torch.float32).to(device)

    elif labels is not None:
      #SupConLoss configuration
      labels = labels.contiguous().view(-1,1)
      if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')
      mask = torch.eq(labels, labels.T).float().to(device)

    else:
      mask = mask.float().to(device)

    contrast_count = features.shape[1] # features: (B, 2, 128)
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # (2B, 128)

    if self.contrast_mode == 'one':
      anchor_feature =features[:, 0] # (B, 128) 
      anchor_count = 1

    elif self.contrast_mode == 'all':
      anchor_feature = contrast_feature # (2B, 128) 
      anchor_count = contrast_count

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),
        self.temperature
    ) # dot product btw anchor and whole features and then divided by temperature for scaling

    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) #logits_max: (2B, 1)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count) 

    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    ) # == ~torch.eye(2B, dtype=bool)

    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask

    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) 

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss =  -(self.temperature / self.base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()

    return loss