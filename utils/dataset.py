#!/usr/bin/env python

import random
import torch as th
import numpy as np

from torch.utils.data.dataloader import default_collate
import torch.utils.data as dat
from torch.nn.utils.rnn import pad_sequence

from .audio import WaveReader

def make_dataloader(train=True,
                    mix_scp=None,
                    ref_scp=None,
                    aux_scp=None,
                    spk_list=None,
                    sample_rate=16000,
                    num_workers=4,
                    chunk_size=32000,
                    batch_size=16,
                    ifvoid=1,
                    df_note=None):
    dataset = Dataset(mix_scp=mix_scp,
                      ref_scp=ref_scp,
                      aux_scp=aux_scp,
                      spk_list=spk_list,
                      sample_rate=sample_rate,
                      ifvoid=ifvoid,
                      df_note=df_note)
    return DataLoader(dataset,
                      train=train,
                      chunk_size=chunk_size,
                      batch_size=batch_size,
                      num_workers=num_workers)


class Dataset(object):
    """
    Per Utterance Loader
    """
    def __init__(self, mix_scp="", ref_scp=None, aux_scp=None, spk_list=None, sample_rate=16000, ifvoid=1, df_note=None):
        self.mix = WaveReader(mix_scp, sample_rate=sample_rate)
        self.ref = WaveReader(ref_scp, sample_rate=sample_rate)
        self.aux = WaveReader(aux_scp, sample_rate=sample_rate)
        self.sample_rate = sample_rate
        self.spk_list = self._load_spk(spk_list)
        self.voidflag = ifvoid

    def _load_spk(self, spk_list_path):
        if spk_list_path is None:
            return []
        lines = open(spk_list_path).readlines()
        new_lines = []
        for line in lines:
            new_lines.append(line.strip())

        return new_lines

    def __len__(self):
        return len(self.mix)

    def find_otherspk(self, key):
        # 在列表中随机找出一条不存在于混合语音中的说话人参考音频
        spk1 = key.split("_")[0].split("-")[0]
        spk2 = key.split("_")[1].split("-")[0]
        
        while True:
            index = random.randint(0, len(self.mix.index_keys)-1)
            newkey = self.mix.index_keys[index]
            newspk = self.spk_list.index(newkey.split("_")[-1].split("-")[0])
            if newspk != spk1 and newspk != spk2:
                newaux = self.aux[newkey]
                break
        return newaux, newspk
    
    def __getitem__(self, index):
        key = self.mix.index_keys[index]
        mix = self.mix[key]
        ref = self.ref[key]
        aux = self.aux[key]
        
        # void = random.randint(1, 3)
        if self.voidflag == 1:
            if index % 4 == 0:
                ref = np.zeros_like(ref)
                aux, spk_idx = self.find_otherspk(key)
            else:
                spk_idx = self.spk_list.index(key.split('_')[-1].split('-')[0])
        elif self.voidflag == 2:
            if ~np.any(ref):
                ref = np.zeros_like(mix)
            spk_idx = self.spk_list.index(key.split('_')[-1].split('-')[0])
        else:
            spk_idx = self.spk_list.index(key.split('_')[-1].split('-')[0])
        return {
            "mix": mix.astype(np.float32),
            "ref": ref.astype(np.float32),
            "aux": aux.astype(np.float32),
            "aux_len": len(aux),
            "spk_idx": spk_idx
        }

class ChunkSplitter(object):
    """
    Split utterance into small chunks
    """
    def __init__(self, chunk_size, train=True, least=16000):
        self.chunk_size = chunk_size
        self.least = least
        self.train = train

    def _make_chunk(self, eg, s):
        """
        Make a chunk instance, which contains:
            "mix": ndarray,
            "ref": [ndarray...]
        """
        chunk = dict()
        chunk["mix"] = eg["mix"][s:s + self.chunk_size]
        chunk["ref"] = eg["ref"][s:s + self.chunk_size]
        chunk["aux"] = eg["aux"]
        chunk["aux_len"] = eg["aux_len"]
        chunk["valid_len"] = int(self.chunk_size)
        chunk["spk_idx"] = eg["spk_idx"]
        return chunk

    def split(self, eg):
        N = eg["mix"].size
        # too short, throw away
        if N < self.least:
            return []
        chunks = []
        # padding zeros
        if N < self.chunk_size:
            P = self.chunk_size - N
            chunk = dict()
            chunk["mix"] = np.pad(eg["mix"], (0, P), "constant")
            chunk["ref"] = np.pad(eg["ref"], (0, P), "constant")
            chunk["aux"] = eg["aux"]
            chunk["aux_len"] = eg["aux_len"]
            chunk["valid_len"] = int(N)
            chunk["spk_idx"] = eg["spk_idx"]
            chunks.append(chunk)
        else:
            # random select start point for training
            s = random.randint(0, N % self.least) if self.train else 0
            while True:
                if s + self.chunk_size > N:
                    break
                chunk = self._make_chunk(eg, s)
                chunks.append(chunk)
                s += self.least
        return chunks


class DataLoader(object):
    """
    Online dataloader for chunk-level
    """
    def __init__(self,
                 dataset,
                 num_workers=4,
                 chunk_size=32000,
                 batch_size=16,
                 train=True):
        self.batch_size = batch_size
        self.train = train
        self.splitter = ChunkSplitter(chunk_size,
                                      train=train,
                                      least=chunk_size // 2)
        # just return batch of egs, support multiple workers
        self.eg_loader = dat.DataLoader(dataset,
                                        batch_size=batch_size // 2,
                                        num_workers=num_workers,
                                        shuffle=train,
                                        collate_fn=self._collate)

    def _collate(self, batch):
        """
        Online split utterances
        """
        chunk = []
        for eg in batch:
            chunk += self.splitter.split(eg)
        return chunk

    def _pad_aux(self, chunk_list):
        lens_list = []
        for chunk_item in chunk_list:
            lens_list.append(chunk_item['aux_len'])
        max_len = np.max(lens_list)
        for idx in range(len(chunk_list)):
            P = max_len - len(chunk_list[idx]["aux"])
            chunk_list[idx]["aux"] = np.pad(chunk_list[idx]["aux"], (0, P), "constant")

        return chunk_list

    def _merge(self, chunk_list):
        """
        Merge chunk list into mini-batch
        """
        N = len(chunk_list)
        if self.train:
            random.shuffle(chunk_list)
        blist = []
        for s in range(0, N - self.batch_size + 1, self.batch_size):
            # padding aux info
            batch = default_collate(self._pad_aux(chunk_list[s:s + self.batch_size]))
            blist.append(batch)
        rn = N % self.batch_size
        return blist, chunk_list[-rn:] if rn else []

    def __iter__(self):
        chunk_list = []
        for chunks in self.eg_loader:
            chunk_list += chunks
            batch, chunk_list = self._merge(chunk_list)
            for obj in batch:
                yield obj
