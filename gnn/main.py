import argparse
import shutil, sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import R2Score
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from gnn.data import CIFData
from gnn.data import mynewCOLLATE, get_train_val_test_loader
from gnn.model import GraphNet

parser = argparse.ArgumentParser(description='GNNTesting')
parser.add_argument('data_options', nargs='+') 
parser.add_argument('--epochs', default=60, type=int)
parser.add_argument('--start', default=0, type=int)
parser.add_argument('--batch-size', default=20, type=int)
parser.add_argument('--lr', default=0.01, type=float) # increase?
parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int)
parser.add_argument('--weight-decay', default=0, type=float)

train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=None, type=float)
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=0.2, type=float)
test_group = parser.add_mutually_exclusive_group() 
test_group.add_argument('--test-ratio', default=0.2, type=float)

parser.add_argument('--atom-fea-len', default=64, type=int)
parser.add_argument('--h-fea-len', default=128, type=int)
parser.add_argument('--n-conv', default=10, type=int, help='num of conv layers')
parser.add_argument('--n-h', default=5, type=int, help='num of hidden layers after pooling')

args = parser.parse_args(sys.argv[1:])

best_mae_error = 1e10

def main():
    global args, best_mae_error
    dataset = CIFData(*args.data_options) 
    collate_fn = mynewCOLLATE
    train_loader, val_loader, test_loader = get_train_val_test_loader(dataset=dataset, collate_fn=collate_fn, batch_size=args.batch_size, train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio, return_test=True)

    sample_data_list = [dataset[i] for i in range(len(dataset))]    
    _, sample_target, _ = mynewCOLLATE(sample_data_list)
    normalizer = Normalizer(sample_target)

    structures, _, _, = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    sputter_fea_len = structures[3].shape[-1]

    model = GraphNet(orig_atom_fea_len, nbr_fea_len, sputter_fea_len, atom_fea_len=args.atom_fea_len, n_conv=args.n_conv, h_fea_len=args.h_fea_len, n_h=args.n_h)
    criterion = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones) 

    for epoch in range(args.start, args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, normalizer) # MARK!!!
        mae_error = validate(val_loader, model, criterion, normalizer)
        if mae_error != mae_error:
            print('exploded NaN')
            sys.exit(1)

        scheduler.step()

        is_best = mae_error < best_mae_error
        best_mae_error = min(mae_error, best_mae_error)
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_mae_error': best_mae_error, 'optimizer': optimizer.state_dict(), 'normalizer': normalizer.state_dict(), 'args': vars(args)}, is_best)

    print('TESTING... TESTING... ')
    best_checkpoint = torch.load('thebestmodel.pth.tar')
    model.load_state_dict(best_checkpoint['state_dict'])
    validate(test_loader, model, criterion, normalizer, test=True)

def train(train_loader, model, criterion, optimizer, epoch, normalizer):
    losses = AverageMeter()

    mae_errors = AverageMeter()
    r2_scores = AverageMeter()

    model.train()

    for i, (input, target, _) in enumerate(train_loader): 
        input_var = (Variable(input[0]), Variable(input[1]), input[2], Variable(input[3]), input[4]) # 3 is sputter features

        target_normed = normalizer.norm(target)
        target_var = Variable(target_normed)

        output = model(*input_var)
        loss = criterion(output, target_var)

        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        r2_score = r2(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu(), target.size(0))
        mae_errors.update(mae_error, target.size(0))
        r2_scores.update(r2_score, target.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})\t'
              'R2 {r2_scores.val:.5f} ({r2_scores.avg:.5f})\t'.format(epoch, i, len(train_loader), loss=losses, mae_errors=mae_errors, r2_scores=r2_scores))

def validate(val_loader, model, criterion, normalizer, test=False):
    losses = AverageMeter()

    mae_errors = AverageMeter()
    r2_scores = AverageMeter()

    if test:
        test_targets, test_preds, test_cif_ids = [], [], []

    model.eval()

    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        with torch.no_grad():
            input_var = (Variable(input[0]), Variable(input[1]), input[2], Variable(input[3]), input[4])
        target_normed = normalizer.norm(target)
        with torch.no_grad():
            target_var = Variable(target_normed)

        output = model(*input_var)
        loss = criterion(output, target_var)

        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        r2_score = r2(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu().item(), target.size(0))
        mae_errors.update(mae_error, target.size(0))
        r2_scores.update(r2_score, target.size(0))

        if test:
            test_pred = normalizer.denorm(output.data.cpu())
            test_target = target
            test_preds += test_pred.view(-1).tolist()
            test_targets += test_target.view(-1).tolist()
            test_cif_ids += batch_cif_ids

        print('Test: [{0}/{1}]\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})\t'
              'R2 {r2_scores.val:.5f} ({r2_scores.avg:.5f})\t'.format(i, len(val_loader), loss=losses, mae_errors=mae_errors, r2_scores=r2_scores))

    if test:
        star_label = '**'
        import csv
        with open('test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets, test_preds):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = '*'
    print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label, mae_errors=mae_errors))
    print(' {star} R2 {r2_scores.avg:.3f}'.format(star=star_label, r2_scores=r2_scores))
    return mae_errors.avg

class Normalizer(object):
    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)
    def norm(self, tensor):
        return (tensor - self.mean) / self.std
    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean
    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}
    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

def mae(prediction, target):
    return torch.mean(torch.abs(target - prediction))
def r2(prediction, target):
    r2score = R2Score()
    return r2score(prediction, target)

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'thebestmodel.pth.tar')

if __name__ == '__main__':
    main()