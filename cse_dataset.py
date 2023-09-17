#!/usr/bin/env python
import os
import random
import torch as th
import numpy as np
import pandas as pd
import soundfile as sf
import argparse

from simpleder.der import DER
from torch.utils.data.dataloader import default_collate
import torch.utils.data as dat
from torch.nn.utils.rnn import pad_sequence

from utils.audio import WaveReader
import pdb


def make_dataloader(train=True,
                    metadata=None,
                    sample_rate=16000,
                    num_workers=4,
                    chunk_size=32000,
                    batch_size=16,
                    aux_setlen=0,
                    sd_strategy=None,
                    rttm_dir=None):
    dataset = Dataset(metadata=metadata,
                      sample_rate=sample_rate,
                      aux_setlen=aux_setlen,
                      sd_strategy=sd_strategy,
                      rttm_dir=rttm_dir)
    return DataLoader(dataset,
                      train=train,
                      chunk_size=chunk_size,
                      batch_size=batch_size,
                      num_workers=num_workers)


def tuplelist_to_vector(tuplelist, length):
    res_vector = np.zeros(length)
    for start, end in tuplelist:
        res_vector[int(start): int(end)] = 1
    return res_vector


def standlist_to_nestdict(standlist):
    nestdict = {}
    for i in range(len(standlist)):
        rttm_tup = (int(standlist[i][1]), int(standlist[i][2]))
        if standlist[i][0] not in nestdict.keys():
            nestdict[standlist[i][0]] = [rttm_tup]
        else:
            nestdict[standlist[i][0]].append(rttm_tup)
    return nestdict


class Dataset(object):
    """
    Per Utterance Loader
    """

    def __init__(self, metadata=None, sample_rate=16000, aux_setlen=0, sd_strategy=None, rttm_dir=None):
        self.sample_rate = sample_rate
        self.aux_setlen = aux_setlen
        self.metadata = pd.read_csv(metadata)
        self.mix = {}
        self.ref = {}
        self.aux = {}
        self.aux_start = {}
        self.overlap = {}
        self.present = {}
        self.single = {}
        self.sd_present = {}
        self.rttm_dict = {}
        self.sdhyp_dict = {}
        self.sdgt_dict = {}
        self.key_to_index = {}
        self.index_keys = []
        self.sd_strategy = sd_strategy
        self.rttm_dir = rttm_dir
        if sd_strategy is not None:
            self._load_rttm()
        self._load_speech()

    def _load_rttm(self):
        # 读取rttm，构造嵌套字典
        for dir in os.listdir(self.rttm_dir):
            if dir[-3:] == "txt":
                continue
            with open(os.path.join(self.rttm_dir, dir), "r") as rttm:
                once_rttm_list = []
                for line in rttm:
                    # key = line.split(" ")[1]
                    key = dir[:-5]
                    start = float(line.split(" ")[3])
                    duration = float(line.split(" ")[4])
                    SD_SPK = line.split(" ")[7]
                    once_rttm_list.append(
                        (SD_SPK, float(start * self.sample_rate), float((start + duration) * self.sample_rate)))
                # {mix_id:[(spk_index,start,end)]}
                self.sdhyp_dict[key] = once_rttm_list.copy()
                # if self.sd_strategy.split("_")[0] == "normal":
                # {mix_id:{spk_index:(start,end)}}
                self.rttm_dict[key] = self._check_rttm(once_rttm_list)

    def _check_rttm(self, rttm_list):
        rttm_dict = {}
        # 检测是否重叠，修改起止点, 合并同一说话人
        for i in range(len(rttm_list) - 1):
            if rttm_list[i][0].split("_") == "top1":
                continue
            # 如果上一条的结束时间大于下一条的开始时间，且不是同一个说话人，说明重叠
            if (rttm_list[i][2] > rttm_list[i + 1][1]) and (rttm_list[i][0] != rttm_list[i + 1][0]):
                # 修改起止时间
                new_end = rttm_list[i + 1][1]
                new_start = rttm_list[i][2]
                rttm_list[i] = (rttm_list[i][0], rttm_list[i][1], new_end)
                rttm_list[i + 1] = (rttm_list[i + 1][0], new_start, rttm_list[i + 1][2])
            # 如果两条相隔不足一秒，且是同一说话人，合并
            if (rttm_list[i + 1][1] - rttm_list[i][2] < 1 * self.sample_rate) and (
                    rttm_list[i][0] == rttm_list[i + 1][0]):
                rttm_list[i + 1] = (rttm_list[i + 1][0], rttm_list[i][1], rttm_list[i + 1][2])
                rttm_list[i] = None

        # 检测是否过短，删除过短的段
        for i in range(len(rttm_list)):
            if rttm_list[i] is not None:
                if (rttm_list[i][2] - rttm_list[i][1]) > 5 * self.sample_rate:
                    rttm_tup = (int(rttm_list[i][1]), int(rttm_list[i][2]))
                    if rttm_list[i][0] not in rttm_dict.keys():
                        rttm_dict[rttm_list[i][0]] = [rttm_tup]
                    else:
                        rttm_dict[rttm_list[i][0]].append(rttm_tup)
        return rttm_dict

    def _load_speech(self):
        # 加载mix ref aux, self.mix self.ref
        index = 0
        for _, row in self.metadata.iterrows():
            key = row["mixture_ID"]
            n_src = int((len(row) - 4) / 3)
            samps, _ = sf.read(row["mixture_path"])
            self.mix[key] = samps
            self.sdgt_dict[key] = []
            self.key_to_index[key] = []
            for i in range(n_src):
                samps, _ = sf.read(row[f"source_{i + 1}_path"])
                key_idx = key + "_" + str(index)
                self.index_keys.append(key_idx)
                self.key_to_index[key].append(key_idx)
                index += 1
                self.ref[key_idx] = samps
                self.mix[key_idx] = self.mix[key]
                self.overlap[key_idx], self.present[key_idx], self.single[key_idx] = self.get_overlap_cond(
                    row[f"lenoverlap"],
                    row[f"source_{i + 1}_len"],
                    row[f"source_{i + 1}_lensingle"],
                    int(row[f"length"]),
                    key, i)

                
                # 测试：mix只使用分开的一段或几段音频
                '''
                indicator = self.overlap[key_idx]
                self.mix[key_idx] = self.mix[key][indicator==1]
                self.ref[key_idx] = self.ref[key_idx][indicator==1]
                self.overlap[key_idx] = self.overlap[key_idx][indicator==1]
                self.present[key_idx] = self.present[key_idx][indicator==1]
                
                len_all_list = row[f"source_{i + 1}_len"][2:-2].split("), (")
                all_tuplelist = [(int(item.split(",")[0]), int(item.split(",")[1])) for item in len_all_list]
                mix = self.mix[key]
                ref = self.ref[key_idx]
                overlap = self.overlap[key_idx]
                present = self.present[key_idx]
                for start, end in all_tuplelist:
                    key_idx = key + "_" + str(index)
                    self.mix[key_idx] = mix[start:end]
                    self.ref[key_idx] = ref[start:end]
                    self.overlap[key_idx] = overlap[start:end]
                    self.present[key_idx] = present[start:end]
                    self.aux[key_idx], self.aux_start[key_idx] = self.get_aux_from_metadata(row[f"source_{i + 1}_lensingle"], self.mix[key])
                    self.sd_present[key_idx] = self.present[key_idx]
                    self.aux[key_idx] = self.get_aux_from_librispeech(key.split("-")[i])
                    if key_idx not in self.index_keys:
                        self.index_keys.append(key_idx)
                        self.key_to_index[key].append(key_idx)
                    index += 1
                '''
                

                if self.sd_strategy is None:
                    self.aux[key_idx], self.aux_start[key_idx] = self.get_aux_from_metadata(
                        row[f"source_{i + 1}_lensingle"], self.mix[key])
                    self.sd_present[key_idx] = self.present[key_idx]
                    self.aux[key_idx] = self.get_aux_from_librispeech(key.split("-")[i])
                else:
                    if self.sd_strategy.split("_")[0] == "normal":
                        self.aux[key_idx], self.aux_start[key_idx] = self.get_aux_from_rttm(
                            row[f"source_{i + 1}_lensingle"], self.mix[key], key)
                    elif self.sd_strategy.split("_")[0] == "max":
                        self.aux[key_idx], self.aux_start[key_idx] = self.get_aux_from_metadata(
                            row[f"source_{i + 1}_lensingle"], self.mix[key])
                    tar_spk = self.align_spk(row[f"source_{i + 1}_lensingle"], self.rttm_dict[key], int(row[f"length"]))
                    nestdict = standlist_to_nestdict(self.sdhyp_dict[key])
                    self.sd_present[key_idx] = tuplelist_to_vector(nestdict[tar_spk], int(row[f"length"]))

    def align_spk(self, lensingle, rttm_dict, length):
        max_sim_score = 0
        tar_spk = "SPEAKER_00"
        for SD_SPK in rttm_dict.keys():
            # 计算两个序列相似性得分，选择最高的
            sim_score = self.len_similarity(rttm_dict[SD_SPK], lensingle, length)
            if sim_score > max_sim_score:
                tar_spk = SD_SPK
                max_sim_score = sim_score
        if max_sim_score == 0:
            tar_spk = SD_SPK
        return tar_spk

    def len_similarity(self, lenrttm, lensingle, length):
        rttm = tuplelist_to_vector(lenrttm, length)
        len_sig_list = lensingle[2:-2].split("), (")
        sig_tuplelist = [(int(item.split(",")[0]), int(item.split(",")[1])) for item in len_sig_list]
        single = tuplelist_to_vector(sig_tuplelist, length)
        lensum = rttm + single
        sim_score = np.sum(lensum >= 2)
        return sim_score

    def get_aux_from_rttm(self, lensingle, mix, key):
        # 根据rttm得出参考音频
        # 因为是测试，所以首先将lensingle作为ground_truth对齐说话人
        rttm_dict = self.rttm_dict[key]
        tar_spk = self.align_spk(lensingle, rttm_dict, len(mix))
        rttm_spk_list = rttm_dict[tar_spk]

        # 选择rttm中说话人的aux
        if self.sd_strategy.split("_")[1] == "longest":
            aux_length = 0
            for start, end in rttm_spk_list:
                if (end - start) > aux_length:
                    aux_length = end - start
                    aux_start = start
                    aux_end = end
            aux = mix[aux_start:aux_end]
            return aux, (aux_start, aux_end)
        elif self.sd_strategy.split("_")[1] == "shortest":
            aux_length = len(mix)
            for start, end in rttm_spk_list:
                if (end - start) < aux_length:
                    aux_length = end - start
                    aux_start = start
                    aux_end = end
            aux = mix[aux_start:aux_end]
            return aux, (aux_start, aux_end)
        elif self.sd_strategy.split("_")[1] == "random":
            aux_start, aux_end = random.sample(rttm_spk_list, 1)[0]
            aux = mix[aux_start:aux_end]
            return aux, (aux_start, aux_end)
        elif self.sd_strategy.split("_")[1] == "clustering":
            choosed_spkid = tar_spk+"_top1"
            for item in self.sdhyp_dict[key]:
                if choosed_spkid == item[0]:
                    start = item[1]
                    end = item[2]
                    aux = mix[int(start):int(end)]
                    return aux, (int(start), int(end))
            aux_length = 0
            for start, end in rttm_spk_list:
                if (end - start) > aux_length:
                    aux_length = end - start
                    aux_start = start
                    aux_end = end
            aux_start += 8000
            aux_end -= 8000
            # aux_start, aux_end = random.sample(rttm_spk_list, 1)[0]
            aux = mix[aux_start:aux_end]
            return aux, (aux_start, aux_end)


    def get_aux_from_metadata(self, lensingle, mix):
        # 得出非重叠且存在的最长区域
        # sig = present - overlap
        # sig[sig < 0] = 0
        len_sig_list = lensingle[2:-2].split("), (")
        aux_length = 0
        aux_start = 0
        aux_end = 0
        for item in len_sig_list:
            start = int(item.split(",")[0])
            end = int(item.split(",")[1])
            if (end - start) > aux_length:
                aux_length = end - start
                aux_start = start
                aux_end = end
        aux = mix[aux_start:aux_end]
        return aux, (aux_start, aux_end)

    def get_aux_from_librispeech(self, spk_id):
        # 对应说话人在LibriSpeech中的路径
        spkpath = os.path.join("../LibriSpeech", "train-clean-100", spk_id)
        files_list = []
        for root, dirs, files in os.walk(spkpath):
            for file in files:
                if file.endswith(".flac"):
                    files_list.append(file)
        aux_flac = random.sample(files_list, 1)[0]
        aux_path = os.path.join(spkpath, aux_flac.split("-")[1], aux_flac.split(".")[0] + "-norm.wav")
        aux, _ = sf.read(aux_path) 
        return aux


    def get_overlap_cond(self, lenov, lenall, lensingle, length, key, i):
        # 根据表格中的(start, end)获取重叠和存在信息
        # len_all=present

        len_sig_list = lensingle[2:-2].split("), (")
        sig_tuplelist = [(int(item.split(",")[0]), int(item.split(",")[1])) for item in len_sig_list]
        single = tuplelist_to_vector(sig_tuplelist, length)

        len_all_list = lenall[2:-2].split("), (")
        all_tuplelist = [(int(item.split(",")[0]), int(item.split(",")[1])) for item in len_all_list]
        present = tuplelist_to_vector(all_tuplelist, length)

        len_ov_list = lenov[2:-2].split("), (")
        ov_tuplelist = [(int(item.split(",")[0]), int(item.split(",")[1])) for item in len_ov_list]
        overlap = tuplelist_to_vector(ov_tuplelist, length)
        for start, end in sig_tuplelist:
            self.sdgt_dict[key].append((str(i), float(start), float(end)))
        return overlap, present, single

    def __len__(self):
        return len(self.index_keys)

    def __getitem__(self, index):
        key_idx = self.index_keys[index]
        key = key_idx.split("_")[0]
        # mix = self.mix[key]
        mix = self.mix[key_idx]
        ref = self.ref[key_idx]
        sd_present = self.sd_present[key_idx]
        overlap = self.overlap[key_idx]
        present = self.present[key_idx]
        aux = self.aux[key_idx]
        if self.aux_setlen > 1:
            if len(aux) > self.aux_setlen:
                aux = aux[:self.aux_setlen]
            else:
                aux = np.pad(aux, (0, self.aux_setlen-len(aux)), mode='constant')
        elif self.aux_setlen < 1 and self.aux_setlen > 0:
            if index + 1 != len(self.index_keys):
                if self.index_keys[index+1].split("_")[0] == key:
                    inter_aux = self.aux[self.index_keys[index+1]]
                else:
                    inter_aux = self.aux[self.index_keys[index-1]]
            else:
                inter_aux = self.aux[self.index_keys[index-1]]
            ov_aux = inter_aux[:int(len(aux)*self.aux_setlen)]
            aux[len(aux) - len(ov_aux):] = 0
            aux = aux + np.pad(ov_aux, (len(aux) - len(ov_aux), 0), mode='constant')

        return {
            "mix": mix.astype(np.float32),
            "ref": ref.astype(np.float32),
            "aux": aux.astype(np.float32),
            "aux_len": len(aux),
            "sd_present": sd_present,
            "overlap": overlap,
            "present": present
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
        chunk["overlap"] = eg["overlap"][s:s + self.chunk_size]
        chunk["present"] = eg["present"][s:s + self.chunk_size]
        chunk["aux"] = eg["aux"]
        chunk["aux_len"] = eg["aux_len"]
        chunk["sd_present"] = eg["sd_present"][s:s + self.chunk_size]
        chunk["valid_len"] = int(self.chunk_size)
        # chunk["spk_idx"] = eg["spk_idx"]
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
            chunk["overlap"] = np.pad(eg["overlap"], (0, P), "constant")
            chunk["present"] = np.pad(eg["present"], (0, P), "constant")
            chunk["aux"] = eg["aux"]
            chunk["aux_len"] = eg["aux_len"]
            chunk["sd_present"] = np.pad(eg["sd_present"], (0, P), "constant")
            chunk["valid_len"] = int(N)
            # chunk["spk_idx"] = eg["spk_idx"]
            chunks.append(chunk)
        else:
            # random select start point for training
            s = 0
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
        # if self.train:
        #     random.shuffle(chunk_list)
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


def trans_rttm_dict(rttm_dict_key):
    # {spk_id:[(start,end)]} -> [(spk_id, start, end)]
    transformed_list = []
    for spk in rttm_dict_key.keys():
        for item in rttm_dict_key[spk]:
            transformed_list.append((spk, float(item[0]), float(item[1])))
    return transformed_list


def aux_error_rate(dataset):
    auxerror = 0
    auxcomplete = 0
    for key_idx in dataset.aux_start.keys():
        start, end = dataset.aux_start[key_idx]
        auxlen = np.zeros_like(dataset.overlap[key_idx])
        auxlen[start: end] = 1
        indicator_ov = auxlen + dataset.overlap[key_idx]
        indicator_sig = auxlen + dataset.single[key_idx]
        aux_length = (end - start)
        if np.sum(indicator_ov >= 2) > 0.05*aux_length:
            auxerror += 1
        complete = np.sum(indicator_sig == 2) / aux_length
        auxcomplete += complete
        # if complete < 0.4:
        #     auxerror += 1
        # else:
        #     auxcomplete += complete
    auxerror = auxerror / len(dataset.aux_start)
    auxcomplete = auxcomplete / len(dataset.aux_start)    
    
    # print("DER={:.3f}".format(error))
    # print("False Alarm={:.3f}".format(fa))
    # print("Miss={:.3f}".format(miss))
    # print("DER_self={:.3f}".format(der))
    print("AER={:.3f}".format(auxerror))
    print("ACOM={:.6f}".format(auxcomplete))


if __name__ == '__main__':
    for mixnum in [2]:
        for ovratio in [20, 30, 40]:
            for sd_strategy in ["normal_longest"]:
                metadata_dir = f"/home/lsh/zyr/LibriCse/Cse{mixnum}Mix/{ovratio}/metadata/mixture_train-100_mix_clean.csv"
                rttm_dir = f"../pyannote-audio/demo/sd3/sd/cse{mixnum}mix{ovratio}ov/"
                print(metadata_dir)
                print(sd_strategy)      
                dataset = Dataset(metadata=metadata_dir,
                                  sample_rate=16000,
                                  sd_strategy=sd_strategy,
                                  rttm_dir=rttm_dir)
                aux_error_rate(dataset)
    '''
    error = 0
    der = 0
    miss = 0
    fa = 0
    N = len(dataset.sdgt_dict)
    for key in dataset.sdgt_dict.keys():
        # error += DER(dataset.sdgt_dict[key], trans_rttm_dict(dataset.rttm_dict[key]))
        size = dataset.mix[key].size
        cost_matrix = np.zeros(size)
        for index in dataset.key_to_index[key]:
            cost = dataset.sd_present[index] - dataset.present[index] 
            # est=1,gt=0(est-gt=1):fa
            fa += np.sum(np.where(cost == 1, 1, 0))/size
            # est=0,gt=1(est-gt=-1):miss
            miss += np.sum(np.where(cost == -1, 1, 0))/size
            cost_matrix += np.abs(cost)
            
        der += np.sum(np.where(cost_matrix > 0, 1, 0))/size
    error = error / N
    der = der / N
    miss = miss / N
    fa = fa / N
    '''
