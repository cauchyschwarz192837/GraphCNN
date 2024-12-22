import argparse
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchmetrics import R2Score
from gnn.data import CIFData
from gnn.data import collate_pool
from gnn.model import GraphNet

parser = argparse.ArgumentParser(description='GNNTest')
parser.add_argument('modelpath')
parser.add_argument('cifpath')
parser.add_argument('-b', '--batch-size', default=20, type=int)

args = parser.parse_args(sys.argv[1:])
model_checkpoint = torch.load(args.modelpath, map_location=lambda storage, loc: storage)
model_args = argparse.Namespace(**model_checkpoint['args'])

best_mae_error = 1e10

def main():
    global args, model_args, best_mae_error

    dataset = CIFData(args.cifpath)
    collate_fn = collate_pool
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=0, collate_fn=collate_fn)

    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    sputter_fea_len = structures[3].shape[-1]

    model = GraphNet(orig_atom_fea_len, nbr_fea_len, sputter_fea_len, atom_fea_len=model_args.atom_fea_len, n_conv=model_args.n_conv, h_fea_len=model_args.h_fea_len, n_h=model_args.n_h, classification=False)

    criterion = nn.MSELoss()
    normalizer = Normalizer(torch.zeros(3))

    validate(test_loader, model, criterion, normalizer, test=True)

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
              'R2 {r2_scores:.5f} ({r2_scores.avg:.5f})\t'.format(i, len(val_loader), loss=losses, mae_errors=mae_errors, r2_scores=r2_scores))

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

if __name__ == '__main__':
    main()