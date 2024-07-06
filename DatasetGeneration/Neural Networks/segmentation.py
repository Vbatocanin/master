import os
import random
import logging
import numpy as np
import torch
import torchvision.transforms as tmf
from PIL import Image as Image
from torch.utils.data import Dataset

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("PIL").setLevel(logging.INFO)

sep = os.sep
CLASS_LABELS = {
    'Background': 0,
    'Caustic': 1,
}
CLASS_RGB = {
    'Background': 0,
    'Caustic': 1
}

num_classes = len(CLASS_LABELS)


class SEGDataset(Dataset):
    def __init__(self, data_dir, image_subdir, label_subdir, label_getter, patch_shape, patch_offset, expand_by):
        self.data_dir = data_dir
        self.image_subdir = image_subdir
        self.label_subdir = label_subdir
        self.label_getter = label_getter
        self.patch_shape = patch_shape
        self.patch_offset = patch_offset
        self.expand_by = expand_by
        self.data = {}
        self.indices = []
        self.load_data()

    def load_data(self):
        files = os.listdir(self.data_dir + '/' + self.image_subdir)
        logging.info(f"Found {len(files)} files in {self.data_dir}/{self.image_subdir}")
        for i, file in enumerate(files):
            self._load_index('SEG', file)
            if (i + 1) % 10 == 0 or (i + 1) == len(files):
                logging.info(f"Processed {i + 1}/{len(files)} images")

    def _load_index(self, dataset_name, file):
        logging.debug(f"Loading image and label for file: {file}")
        img_obj = Image.open(self.data_dir + '/' + self.image_subdir + '/' + file).convert('RGB')
        ground_truth = Image.open(self.data_dir + '/' + self.label_subdir + '/' + file).convert('L')

        self.data[file] = {'image': img_obj, 'gt': ground_truth}
        for chunk_ix in get_chunk_indexes(img_obj.size, self.patch_shape, self.patch_offset):
            self.indices.append([dataset_name, file] + [chunk_ix])
        logging.debug(f"Loaded image and label for file: {file}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        dname, file, row_from, row_to, col_from, col_to = self.indices[index]
        input_size = tuple(map(sum, zip(self.patch_shape, self.expand_by)))

        arr = np.array(self.data[file]['image'])
        gt = np.array(self.data[file]['gt'])[row_from:row_to, col_from:col_to] / 255

        p, q, r, s, pad = expand_and_mirror_patch(arr.shape[:2], [row_from, row_to, col_from, col_to], self.expand_by)
        arr3 = np.zeros((input_size[0], input_size[1], 3), dtype=np.uint8)
        arr3[:, :, 0] = np.pad(arr[p:q, r:s, 0], pad, 'reflect')
        arr3[:, :, 1] = np.pad(arr[p:q, r:s, 1], pad, 'reflect')
        arr3[:, :, 2] = np.pad(arr[p:q, r:s, 2], pad, 'reflect')

        if random.uniform(0, 1) <= 0.5:
            arr3 = np.flip(arr3, 0)
            gt = np.flip(gt, 0)

        if random.uniform(0, 1) <= 0.5:
            arr3 = np.flip(arr3, 1)
            gt = np.flip(gt, 1)

        arr3 = self.transforms(arr3)
        return {'input': arr3, 'label': gt.copy()}

    @property
    def transforms(self):
        return tmf.Compose([tmf.ToPILImage(), tmf.ToTensor()])


class SEGTrainer:
    def __init__(self, model, dataloader, device, grad_accum):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.grad_accum = grad_accum
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = torch.nn.CrossEntropyLoss()

    def run(self, epochs):
        for epoch in range(epochs):
            self.train_epoch()
            logging.info(f"Epoch {epoch + 1}/{epochs} completed.")

    def train_epoch(self):
        self.model.train()
        total_batches = len(self.dataloader)
        for i, batch in enumerate(self.dataloader):
            inputs = batch['input'].to(self.device).float()
            labels = batch['label'].to(self.device).long()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()

            if (i + 1) % self.grad_accum == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                logging.info(f"Batch {i + 1}/{total_batches}, Loss: {loss.item()}")


def get_chunk_indexes(shape, patch_shape, patch_offset):
    indexes = []
    for i in range(0, shape[0] - patch_shape[0] + 1, patch_offset[0]):
        for j in range(0, shape[1] - patch_shape[1] + 1, patch_offset[1]):
            indexes.append((i, i + patch_shape[0], j, j + patch_patch_shape[1]))
    return indexes


def expand_and_mirror_patch(shape, chunk_ix, expand_by):
    p, q, r, s = chunk_ix
    pad_top = min(p, expand_by[0])
    pad_bottom = min(shape[0] - q, expand_by[0])
    pad_left = min(r, expand_by[1])
    pad_right = min(shape[1] - s, expand_by[1])

    return p - pad_top, q + pad_bottom, r - pad_left, s + pad_right, ((pad_top, pad_bottom), (pad_left, pad_right))
