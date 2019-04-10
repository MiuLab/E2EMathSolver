import collections
import numpy as np
import torch
import torch.utils.data.dataloader
from torch.autograd import Variable
from tqdm import tqdm
import pdb


class TorchBase():
    def _run_iter(self, batch, training):
        pass

    def _predict_batch(self, batch):
        pass

    def _run_epoch(self, dataloader, training):
        # set model training/evaluation mode
        self._model.train(training)

        # run batches for train
        loss = 0

        # init metric_scores
        # metric_scores = {}
        # for metric in self._metrics:
        #     metric_scores[metric] = 0

        for batch in tqdm(dataloader):
            outputs, batch_loss = \
                self._run_iter(batch, training)

            if training:
                self._optimizer.zero_grad()
                batch_loss.backward()
                self._optimizer.step()

            loss += batch_loss.item()
            # for metric, func in self._metrics.items():
            #     metric_scores[metric] += func(

        # calculate averate loss
        loss /= (len(dataloader) + 1e-6)

        epoch_log = {}
        epoch_log['loss'] = float(loss)
        print('loss=%f\n' % loss)
        return epoch_log

    def __init__(self,
                 learning_rate=1e-3, batch_size=10,
                 n_epochs=10, valid=None,
                 reg_constant=0.0,
                 device=None):

        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._n_epochs = n_epochs
        self._valid = valid
        self._reg_constant = reg_constant
        self._epoch = 0
        if device is not None:
            self._device = torch.device(device)
        else:
            self._device = torch.device('cuda:0' if torch.cuda.is_available()
                                        else 'cpu')

    def fit_dataset(self, data, callbacks=[]):
        # Start the training loop.
        while self._epoch < self._n_epochs:

            # train and evaluate train score
            print('training %i' % self._epoch)
            dataloader = torch.utils.data.DataLoader(
                data,
                batch_size=self._batch_size,
                shuffle=True,
                collate_fn=skip_list_collate,
                num_workers=0)
            # train epoch
            log_train = self._run_epoch(dataloader, True)

            # evaluate valid score
            if self._valid is not None and len(self._valid) > 0:
                print('evaluating %i' % self._epoch)
                dataloader = torch.utils.data.DataLoader(
                    self._valid,
                    batch_size=self._batch_size,
                    shuffle=True,
                    collate_fn=skip_list_collate,
                    num_workers=1)
                # evaluate model
                log_valid = self._run_epoch(dataloader, False)
            else:
                log_valid = {}

            for callback in callbacks:
                callback.on_epoch_end(log_train, log_valid, self)

            self._epoch += 1

    def predict_dataset(self, data,
                        batch_size=None,
                        predict_fn=None,
                        progress_bar=True):
        if batch_size is None:
            batch_size = self._batch_size
        if predict_fn is None:
            predict_fn = self._predict_batch

        # set model to eval mode
        self._model.eval()

        # make dataloader
        dataloader = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=skip_list_collate,
            num_workers=1)

        ys_ = []
        dataloader = tqdm(dataloader) if progress_bar else dataloader
        for batch in dataloader:
            with torch.no_grad():
                batch_y_ = predict_fn(batch)
                ys_ += batch_y_

        return ys_

    def save(self, path):
        torch.save({
            'epoch': self._epoch + 1,
            'model': self._model.state_dict(),
            # 'optimizer': self._optimizer.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self._epoch = checkpoint['epoch']
        self._model.load_state_dict(checkpoint['model'])
        # self._optimizer.load_state_dict(checkpoint['optimizer'])


numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


def skip_list_collate(batch):
    """
    Puts each data field into a tensor with outer dimension batch size.
    Do not collect list recursively.
    """
    if torch.is_tensor(batch[0]):
        out = None
        if torch.utils.data.dataloader._use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif type(batch[0]).__module__ == 'numpy':
        elem = batch[0]
        if type(elem).__name__ == 'ndarray':
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], (str, bytes)):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: skip_list_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        # do not collate list recursively
        return batch

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))
