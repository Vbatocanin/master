import argparse
import os
import torch
from torch.utils.data import DataLoader
from segmentation import SEGDataset, SEGTrainer
from models import UNet


def same(x): return x


SEG_DATASET_PARAMETERS = {
    'data_dir': 'SEG',
    'image_subdir': 'Picture_Caustic',
    'label_subdir': 'Seg_Processed',
    'split_dir':  'splits',
    'label_getter': same,
    'patch_shape': (388, 388),
    'patch_offset': (350, 350),
    'expand_by': (184, 184)
}


def main(args):
    dataset = SEGDataset(**SEG_DATASET_PARAMETERS)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = UNet(num_channels=3, num_classes=2, reduce_by=args.model_scale)
    trainer = SEGTrainer(model=model, dataloader=dataloader,
                         device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                         grad_accum=args.grad_accum)

    trainer.run(epochs=args.epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ms', '--model_scale', default=1, type=float, help='model scale (double)')
    parser.add_argument('-nw', '--num_workers', default=1, type=int, help='number of worker threads (int)')
    parser.add_argument('-b', '--batch_size', default=4, type=int, help='batch size (int)')
    parser.add_argument('-gi', '--grad_accum', default=1, type=int, help='gradient accumulation (int)')
    parser.add_argument('-ep', '--epochs', default=251, type=int, help='number of epochs to train (int)')
    parser.add_argument('-ph', '--phase', default='train', type=str, help='phase (train or test)')


    args = parser.parse_args()
    main(args)
