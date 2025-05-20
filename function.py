import os
import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def split_image(image_tensor, n_blocks):
    height, width = image_tensor.shape[-2:]
    block_height = height // n_blocks
    block_width = width // n_blocks
    blocks = []
    for i in range(n_blocks):
        for j in range(n_blocks):
            start_h = i * block_height
            end_h = start_h + block_height
            start_w = j * block_width
            end_w = start_w + block_width
            block = image_tensor[:, :, start_h:end_h, start_w:end_w]
            blocks.append(block)
    return blocks

def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=10):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""

    decay_rate = 0.8 ** (epoch // lr_decay_epoch)
    if epoch % lr_decay_epoch == 0:
        print('decay_rate is set to {}'.format(decay_rate))

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

    return optimizer

def train(model, train_loader, test_loader, criterion, optimizer, n_blocks, min_epochs=100):
    model.to(device)
    model.train()

    best_srcc = 0.0
    best_plcc = 0.0
    no_improvement_count = 0
    print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC')
    epoch = 0
    while no_improvement_count < 20 or epoch < min_epochs:
        total_loss = 0
        pred_scores = []
        gt_scores = []
        for images, quality_scores in train_loader:
            images = images.to(device)
            quality_scores = quality_scores.to(device)
            final_scores = []
            for img in images:
                blocks = split_image(img.unsqueeze(0), n_blocks=n_blocks)
                blocks_tensor = torch.cat(blocks, dim=0)
                final_score = model(blocks_tensor).to(device)
                final_scores.append(final_score)
            final_scores_tensor = torch.stack(final_scores).to(device)

            optimizer.zero_grad()
            loss = criterion(final_scores_tensor, quality_scores.view(-1).float())  # 调整目标尺寸
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        pred_scores.extend(final_scores_tensor.cpu().detach().numpy().astype(np.float64))
        gt_scores.extend(quality_scores.cpu().numpy())
        train_srcc, _ = spearmanr(pred_scores, gt_scores)

        test_srcc, test_plcc = test(model, test_loader, n_blocks)
        optimizer = exp_lr_scheduler(optimizer, epoch)  # 调用自定义的学习率调度器
        if test_srcc > best_srcc:
            best_srcc = test_srcc
            best_plcc = test_plcc
            # torch.save(model.state_dict(), os.path.join("test-result", f'{dataset_name}_best_model.pth'))
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        print('%d\t  %4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
              (epoch + 1, total_loss / len(train_loader), train_srcc, test_srcc, test_plcc))
        epoch += 1

    print('Best test SRCC %f, PLCC %f' % (best_srcc, best_plcc))
    return best_srcc, best_plcc

def test(model, test_loader, n_blocks):
    model.eval()
    pred_scores = []
    gt_scores = []

    with torch.no_grad():
        for images, quality_scores in test_loader:
            images = images.to(device)
            final_scores = []
            for img in images:
                blocks = split_image(img.unsqueeze(0), n_blocks=n_blocks)
                blocks_tensor = torch.cat(blocks, dim=0)
                final_score = model(blocks_tensor).to(device)
                final_scores.append(final_score)
            final_scores_tensor = torch.stack(final_scores).to(device)

            pred_scores.extend(final_scores_tensor.cpu().detach().numpy().astype(np.float64))
            gt_scores.extend(quality_scores.cpu().numpy())

    test_srcc, _ = spearmanr(pred_scores, gt_scores)
    test_plcc, _ = pearsonr(pred_scores, gt_scores)
    return test_srcc, test_plcc