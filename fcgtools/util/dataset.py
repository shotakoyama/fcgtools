import torch

class Dataset(torch.utils.data.Dataset):

    def __init__(self, src_data, trg_data = None):
        self.src_data = src_data
        self.trg_data = trg_data

        self.src_lengths = torch.tensor([len(sent) for sent in src_data])

        if self.trg_data is not None:
            self.trg_lengths = torch.tensor([len(sent) for sent in trg_data])

        self.lengths = self.src_lengths

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, index):
        src = self.src_data[index]

        if self.trg_data is None:
            dct = {'index': index, 'source': src}
        else:
            trg = self.trg_data[index]
            dct = {'index': index, 'source': src, 'target': trg}

        return dct

