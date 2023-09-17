#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2023/6/28 15:18
# @Author : EASE
# @Version：V 0.1
# @File : cse_test.py
# @desc :

# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2023/4/11 14:24
# @Author : EASE
# @Version：V 0.1
# @File : evaluate_dev
# @desc :

import os
import pprint
import argparse
import sys

sys.path.append("../")

from cse_dataset import make_dataloader
from utils.logger import get_logger
from utils.load_obj import load_obj
from utils.timer import Timer
from nnet.spex_plus import SpEx_Plus
from hparam import Hparam
import simpleder

# TSEV = True

import torch as th
import pandas as pd
import pdb

logger = get_logger(__name__)


class Reporter(object):
    """
    A progress reporter to record the loss for each batch
    """

    def __init__(self, logger, period=100):
        self.period = period
        self.logger = logger
        self.loss = []
        self.energy = []
        self.sdr = []

        self.mix_energy = []
        self.sig_energy = []
        self.mix_sdr = []
        self.sig_sdr = []

        # self.PEN = []
        # self.NSN = []
        self.PEN_M = []
        self.PEN_S = []
        self.NSN_M = []
        self.NSN_S = []
        # self.DN = []
        self.TA_M = []
        self.TA_S = []
        self.TP_M = []
        self.TP_S = []
        self.nsample = []
        self.timer = Timer()

    def add(self, snr_loss, energy, sdr,
            mix_energy, sig_energy, mix_sdr, sig_sdr,
            PEN_M, PEN_S, NSN_M, NSN_S, TA_M, TA_S, TP_M, TP_S, n):
        self.loss.append(snr_loss)
        self.energy.append(energy)
        self.sdr.append(sdr)
        self.mix_energy.append(mix_energy)
        self.sig_energy.append(sig_energy)
        self.mix_sdr.append(mix_sdr)
        self.sig_sdr.append(sig_sdr)
        self.PEN_M.append(PEN_M)
        self.PEN_S.append(PEN_S)
        self.NSN_M.append(NSN_M)
        self.NSN_S.append(NSN_S)
        self.TA_M.append(TA_M)
        self.TA_S.append(TA_S)
        self.TP_M.append(TP_M)
        self.TP_S.append(TP_S)
        self.nsample.append(n)
        N = len(self.loss)
        if not N % self.period:
            avg_loss = sum(self.loss[-self.period:]) / self.period
            avg_energy = sum(self.energy[-self.period:]) / (
                    self.period - self.energy[-self.period:].count(th.tensor(0)) + 1)
            avg_sdr = sum(self.sdr[-self.period:]) / (self.period - self.sdr[-self.period:].count(th.tensor(0)))

            avg_mix_energy = sum(self.mix_energy[-self.period:]) / (
                    self.period - self.mix_energy[-self.period:].count(th.tensor(0)) + 1)
            avg_sig_energy = sum(self.sig_energy[-self.period:]) / (
                    self.period - self.sig_energy[-self.period:].count(th.tensor(0)) + 1)
            avg_mix_sdr = sum(self.mix_sdr[-self.period:]) / (
                    self.period - self.mix_sdr[-self.period:].count(th.tensor(0)) + 1)
            avg_sig_sdr = sum(self.sig_sdr[-self.period:]) / (
                    self.period - self.sig_sdr[-self.period:].count(th.tensor(0)) + 1)

            PER = (sum(self.PEN_M[-self.period:]) + sum(self.PEN_S[-self.period:])) / (
                    sum(self.TA_M[-self.period:]) + sum(self.TA_S[-self.period:]) + 1) * 100.0
            NSR = (sum(self.NSN_M[-self.period:]) + sum(self.NSN_S[-self.period:])) / (
                    sum(self.TP_M[-self.period:]) + sum(self.TP_S[-self.period:]) + 1) * 100.0
            PER_M = sum(self.PEN_M[-self.period:]) / (sum(self.TA_M[-self.period:]) + 1) * 100.0
            PER_S = sum(self.PEN_S[-self.period:]) / (sum(self.TA_S[-self.period:]) + 1) * 100.0
            NSR_M = sum(self.NSN_M[-self.period:]) / (sum(self.TP_M[-self.period:]) + 1) * 100.0
            NSR_S = sum(self.NSN_S[-self.period:]) / (sum(self.TP_S[-self.period:]) + 1) * 100.0
            ExER = (sum(self.PEN_M[-self.period:]) + sum(self.PEN_S[-self.period:]) + 
                    sum(self.NSN_M[-self.period:]) + sum(self.NSN_S[-self.period:])) / (
                    sum(self.TA_M[-self.period:]) + sum(self.TA_S[-self.period:]) +
                    sum(self.TP_M[-self.period:]) + sum(self.TP_S[-self.period:]) + 1) * 100.0
            self.logger.info("Processed {:d} batches "
                             "avg = {:+.4f}, energy = {:+.4f}, sdr = {:+.4f},\n"
                             "mix_energy = {:+.4f}, sig_energy = {:+.4f}, mix_sdr = {:+.4f}, sig_sdr = {:+.4f},\n"
                             "PER = {:.2f}, NSR= {:.2f},\n"
                             "PER_M = {:.2f}, PER_S = {:.2f}, NSR_M = {:.2f}, NSR_S = {:.2f}, ExER = {:.2f}"
                             "".format(N, avg_loss, avg_energy, avg_sdr,
                                       avg_mix_energy, avg_sig_energy, avg_mix_sdr, avg_sig_sdr,
                                       PER, NSR, PER_M, PER_S, NSR_M, NSR_S, ExER
                                       ))

    def report(self, details=False):
        N = len(self.loss)
        return {
            "loss": sum(self.loss) / N,
            "energy": sum(self.energy) / (N - self.energy.count(th.tensor(0)) + 1),
            "sdr": sum(self.sdr) / (N - self.sdr.count(th.tensor(0))),
            "mix_energy": sum(self.mix_energy) / (N - self.mix_energy.count(th.tensor(0)) + 1),
            "sig_energy": sum(self.sig_energy) / (N - self.sig_energy.count(th.tensor(0)) + 1),
            "mix_sdr": sum(self.mix_sdr) / (N - self.mix_sdr.count(th.tensor(0)) + 1),
            "sig_sdr": sum(self.sig_sdr) / (N - self.sig_sdr.count(th.tensor(0)) + 1),
            "PER": (sum(self.PEN_M) + sum(self.PEN_S)) / (sum(self.TA_M) + sum(self.TA_S) + 1) * 100.0,
            "NSR": (sum(self.NSN_M) + sum(self.NSN_S)) / (sum(self.TP_M) + sum(self.TP_S) + 1) * 100.0,
            "PER_M": sum(self.PEN_M) / (sum(self.TA_M) + 1) * 100.0,
            "PER_S": sum(self.PEN_S) / (sum(self.TA_S) + 1) * 100.0,
            "NSR_M": sum(self.NSN_M) / (sum(self.TP_M) + 1) * 100.0,
            "NSR_S": sum(self.NSN_S) / (sum(self.TP_S) + 1) * 100.0,
            "ExER" : (sum(self.PEN_M) + sum(self.PEN_S) + sum(self.NSN_M) + sum(self.NSN_S)) / (
                    sum(self.TA_M) + sum(self.TA_S) + sum(self.TP_M) + sum(self.TP_S) + 1) * 100.0,
            "cost": self.timer.elapsed(),
            "batches": N
        }


class Trainer(object):
    def __init__(self,
                 nnet,
                 checkpoint="checkpoint",
                 gpuid=0,
                 logging_period=200):
        if not th.cuda.is_available():
            raise RuntimeError("CUDA device unavailable...exist")
        if not isinstance(gpuid, tuple):
            gpuid = (gpuid,)
        self.device = th.device("cuda:{}".format(gpuid[0]))
        self.gpuid = gpuid
        if checkpoint and not os.path.exists(checkpoint):
            raise RuntimeError("checkpoint is not exist")
        self.checkpoint = checkpoint
        self.logger = get_logger(os.path.join(checkpoint, args.log_name), file=True)
        self.logging_period = logging_period

        resume = os.path.join(checkpoint, "best.pt.tar")
        if not os.path.exists(resume):
            raise FileNotFoundError(
                "Could not find resume checkpoint: {}".format(resume))
        cpt = th.load(resume, map_location="cpu")
        self.cur_epoch = cpt["epoch"]
        self.logger.info("Resume from checkpoint {}: epoch {:d}".format(
            resume, self.cur_epoch))
        # load nnet
        nnet.load_state_dict(cpt["model_state_dict"])
        self.nnet = nnet.to(self.device)
        self.optimizer = th.optim.Adam(self.nnet.parameters())
        self.optimizer.load_state_dict(cpt["optim_state_dict"])
        self.num_params = sum(
            [param.nelement() for param in nnet.parameters()]) / 10.0 ** 6
        # logging
        # self.logger.info("Model summary:\n{}".format(nnet))
        self.logger.info("Loading model to GPUs:{}, #param: {:.2f}M".format(
            gpuid, self.num_params))
        self.logger.info("test set:{}".format(args.metadata_dir))
        self.logger.info("sd_strategy:{}".format(args.sd_strategy))
        self.logger.info("rttm:{}".format(args.rttm_dir))
        self.logger.info("description:{}".format(args.description))

    def compute_loss(self, egs):
        raise NotImplementedError

    def eval(self, data_loader):
        self.logger.info("Set eval mode...")
        self.nnet.eval()
        reporter = Reporter(self.logger, period=self.logging_period)

        with th.no_grad():
            for egs in data_loader:
                egs = load_obj(egs, self.device)
                snr_loss, energy, sdr, \
                mix_energy, sig_energy, mix_sdr, sig_sdr, \
                PEN_M, PEN_S, NSN_M, NSN_S, \
                TA_M, TA_S, TP_M, TP_S, N = self.compute_loss(egs)
                reporter.add(snr_loss.item(), energy.item(), sdr.item(),
                             mix_energy.item(), sig_energy.item(), mix_sdr.item(), sig_sdr.item(),
                             PEN_M, PEN_S, NSN_M, NSN_S, TA_M, TA_S, TP_M, TP_S, N)
        return reporter.report(details=False)

    def run(self, dev_loader):
        # avoid alloc memory from gpu0
        with th.cuda.device(self.gpuid[0]):
            # check if save is OK
            cv = self.eval(dev_loader)
            self.logger.info("SUM:"
                             "avg = {:+.4f}, energy = {:+.4f}, sdr = {:+.4f},\n"
                             "mix_energy = {:+.4f}, sig_energy = {:+.4f}, mix_sdr = {:+.4f}, sig_sdr = {:+.4f},\n"
                             "PER = {:.2f}, NSR= {:.2f},\n"
                             "PER_M = {:.2f}, PER_S = {:.2f}, NSR_M = {:.2f}, NSR_S = {:.2f}, ExER = {:.2f}"
                             "".format(cv["loss"], cv["energy"], cv["sdr"],
                                       cv["mix_energy"], cv["sig_energy"], cv["mix_sdr"], cv["sig_sdr"],
                                       cv["PER"], cv["NSR"], cv["PER_M"], cv["PER_S"], cv["NSR_M"], cv["NSR_S"], cv["ExER"]
                                       ))


class SiSnrTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(SiSnrTrainer, self).__init__(*args, **kwargs)

    def getBinaryTensor(self, imgTensor, boundary=0):
        # 若tensor中元素大于0时则置0，否则置1
        one = th.ones_like(imgTensor)
        zero = th.zeros_like(imgTensor)
        return th.where(imgTensor > boundary, zero, one)

    def getPositiveTensor(self, imgTensor, boundary=0):
        zero = th.zeros_like(imgTensor)
        return th.where(imgTensor > boundary, imgTensor, zero)

    def l2norm(self, mat, keepdim=False):
        return th.norm(mat, dim=-1, keepdim=keepdim)

    def sisdr(self, x, s, m, eps=1e-8):
        if x.shape != s.shape:
            raise RuntimeError(
                "Dimention mismatch when calculate si-snr, {} vs {}".format(
                    x.shape, s.shape))
        # est
        x_zm = x - th.mean(x, dim=-1, keepdim=True)
        # ref
        s_zm = s - th.mean(s, dim=-1, keepdim=True)
        # mix
        m_zm = m - th.mean(m, dim=-1, keepdim=True)
        t = th.sum(
            x_zm * s_zm, dim=-1,
            keepdim=True) * s_zm / (self.l2norm(s_zm, keepdim=True) ** 2 + eps)
        mix_t = th.sum(m_zm * s_zm, dim=-1, keepdim=True) * s_zm / (self.l2norm(s_zm, keepdim=True) ** 2 + eps)
        # 修改sisdr loss，使之可以反映目标为0时est的质量
        BinaryTensor = self.getBinaryTensor(self.l2norm(s_zm))
        sisdr = 20 * th.log10(eps + self.l2norm(t) / (self.l2norm(x_zm - t) + eps))
        mix_sisdr = 20 * th.log10(eps + self.l2norm(mix_t) / (self.l2norm(m_zm - mix_t) + eps))
        # xsisdr = sisdr * (-1 * BinaryTensor + 1) - BinaryTensor * th.log10(
        #     self.l2norm(x_zm) ** 2 + (self.l2norm(m_zm) ** 2) * 0.001) * 10
        avgsisdr = sisdr * (-1 * BinaryTensor + 1) - BinaryTensor * th.log10(self.l2norm(x_zm) ** 2 + eps) * 10
        sisdri = sisdr - mix_sisdr
        # 将sisdri改为energy，表示等效der，正值表示目标说话人存在
        # return avgsisdr, th.log10(self.l2norm(x_zm) ** 2 + eps)*10, BinaryTensor
        return avgsisdr, sisdri, BinaryTensor

    def mask_by_length(self, xs, lengths, fill=0):
        assert xs.size(0) == len(lengths)
        ret = xs.data.new(*xs.size()).fill_(fill)
        for i, l in enumerate(lengths):
            ret[i, :l] = xs[i, :l]
        return ret

    def mask_by_sdpresent(self, est, sd_present):
        zero = th.zeros_like(est)
        ret = est * sd_present + zero * (-1 * sd_present + 1)
        return ret

    def compute_loss(self, egs):
        sd_present = egs["sd_present"]
        refs = egs["ref"]
        mixs = egs["mix"]
        ests, ests2, ests3, indicator = th.nn.parallel.data_parallel(
            self.nnet, (egs["mix"], egs["aux"], egs["aux_len"]), device_ids=self.gpuid)
        sd_present = egs["sd_present"]
        if args.sd_strategy == "max":
            ests = self.mask_by_sdpresent(ests, sd_present)
            ests2 = self.mask_by_sdpresent(ests2, sd_present)
            ests3 = self.mask_by_sdpresent(ests3, sd_present)
        overlap = -1 * self.getBinaryTensor(self.l2norm(egs["overlap"])) + 1
        present = self.getBinaryTensor(self.l2norm(egs["present"]))

        ## P x N
        N = egs["mix"].size(0)
        valid_len = egs["valid_len"]
        ests = self.mask_by_length(ests, valid_len)
        ests2 = self.mask_by_length(ests2, valid_len)
        ests3 = self.mask_by_length(ests3, valid_len)
        refs = self.mask_by_length(refs, valid_len)

        snr1, sisdri1, binarytensor1 = self.sisdr(ests, refs, mixs)
        snr2, sisdri2, binarytensor2 = self.sisdr(ests2, refs, mixs)
        snr3, sisdri3, binarytensor3 = self.sisdr(ests3, refs, mixs)

        # SDR Dist
        snr_mean = 0.8 * snr1 + 0.1 * snr2 + 0.1 * snr3
        sisdri_mean = 0.8 * sisdri1 + 0.1 * sisdri2 + 0.1 * sisdri3
        # getPositiveTensor

        energy = th.sum(snr_mean * binarytensor1)
        sdr = th.sum(snr_mean * (-1 * binarytensor1 + 1))

        mix_energy = th.sum(snr_mean * binarytensor1 * overlap)
        sig_energy = th.sum(snr_mean * binarytensor1 * (-1 * overlap + 1))
        # pos_sdr = self.getPositiveTensor(snr_mean)
        # mix_sdr = th.sum(pos_sdr * (-1 * binarytensor1 + 1) * overlap)
        mix_sdr = th.sum(snr_mean * (-1 * binarytensor1 + 1) * overlap)
        # 暂时将mix_sdr替换为mix_sdri
        # mix_sdr = th.sum(sisdri_mean * (-1 * binarytensor1 + 1) * overlap)
        # mix_sdr = th.where(mix_sdri > 0, mix_sdri, th.zeros_like(mix_sdri))
        sisi_TP_M = th.sum((snr_mean * (-1 * binarytensor1 + 1) * overlap) > 0)
        sig_sdr = th.sum(snr_mean * (-1 * binarytensor1 + 1) * (-1 * overlap + 1))
        TA_M = th.sum(binarytensor1 * overlap)
        TA_S = th.sum(binarytensor1 * (-1 * overlap + 1))
        TP_M = th.sum((-1 * binarytensor1 + 1) * overlap)
        TP_S = th.sum((-1 * binarytensor1 + 1) * (-1 * overlap + 1))
        if energy.float() != 0:
            energy = energy / th.sum(binarytensor1)
        if sdr.float() != 0:
            sdr = sdr / th.sum(-1 * binarytensor1 + 1)
        if TA_M != 0:
            mix_energy = mix_energy / TA_M
        if TA_S != 0:
            sig_energy = sig_energy / TA_S
        if TP_M != 0:
            mix_sdr = mix_sdr / TP_M
            # mix_sdr = mix_sdr / sisi_TP_M
        if TP_S != 0:
            sig_sdr = sig_sdr / TP_S
        snr_loss = (-0.8 * th.sum(snr1) - 0.1 * th.sum(snr2) - 0.1 * th.sum(snr3)) / N

        # NSR NDR
        # snr_mean 改为 sisdri_mean
        PEN_M = th.sum((snr_mean * binarytensor1 * overlap) < 0)
        PEN_S = th.sum((snr_mean * binarytensor1 * (-1 * overlap + 1)) < 0)
        NSN_M = th.sum((snr_mean * (-1 * binarytensor1 + 1) * overlap) < 0)
        NSN_S = th.sum((snr_mean * (-1 * binarytensor1 + 1) * (-1 * overlap + 1)) < 0)

        '''
        # energy 为正表示有音频存在，binarytensor1=1指标签为没有音频 
        # 若在本该没有音频的段中出现了音频，说明发生了false alarm错误
        PEN_M = th.sum(sisdri_mean[th.where(binarytensor1 * overlap)] > -30) 
        PEN_S = th.sum(sisdri_mean[th.where(binarytensor1 * (-1 * overlap + 1))] > -30)
        # binarytensor1=0指标签为有音频
        # 若在本该有音频的段中没有出现音频，说明发生了miss错误
        NSN_M = th.sum(sisdri_mean[th.where((-1 * binarytensor1 + 1) * overlap)] < -30)
        NSN_S = th.sum(sisdri_mean[th.where((-1 * binarytensor1 + 1) * (-1 * overlap + 1))] < -30)
        '''

        return snr_loss, energy, sdr, \
               mix_energy, sig_energy, mix_sdr, sig_sdr, \
               PEN_M, PEN_S, NSN_M, NSN_S, TA_M, TA_S, TP_M, TP_S, N


def run(args, hp):
    gpus = str(hp.gpus)
    gpuids = tuple(map(int, gpus.split(",")))

    L1 = int(hp.L1 * hp.sample_rate)
    L2 = int(hp.L2 * hp.sample_rate)
    L3 = int(hp.L3 * hp.sample_rate)
    nnet = SpEx_Plus(L1=L1,
                     L2=L2,
                     L3=L3,
                     N=hp.N,
                     B=hp.B,
                     O=hp.O,
                     P=hp.P,
                     Q=hp.Q,
                     num_spks=hp.num_spks,
                     # ft_type=hp.dtype,
                     spk_embed_dim=hp.spk_embed_dim,
                     causal=hp.causal)
    trainer = SiSnrTrainer(nnet,
                           gpuid=gpuids,
                           checkpoint=args.checkpoint,
                           logging_period=500)

    chunk_size = hp.chunk_size * hp.sample_rate
    if args.sd_strategy == "None":
        sd_strategy = None
        rttm_dir = None
    else:
        sd_strategy = args.sd_strategy
        rttm_dir = args.rttm_dir
    if args.description.isdigit():
        description = float(args.description)
    else:
        description = 0
    dev_loader = make_dataloader(train=False,
                                 metadata=args.metadata_dir,
                                 sample_rate=hp.sample_rate,
                                 batch_size=hp.batch_size,
                                 chunk_size=chunk_size,
                                 num_workers=hp.num_workers,
                                 aux_setlen=description,
                                 sd_strategy=sd_strategy,
                                 rttm_dir=rttm_dir)

    trainer.run(dev_loader)


def diarization_precision(args, hp):
    chunk_size = hp.chunk_size * hp.sample_rate
    sd_loader = make_dataloader(train=False,
                                metadata=args.metadata_dir,
                                sample_rate=hp.sample_rate,
                                batch_size=hp.batch_size,
                                chunk_size=chunk_size,
                                num_workers=hp.num_workers)
                                # sd_strategy=args.sd_strategy,
                                # rttm_dir=args.rttm_dir)
    batchnum = 0
    for sd_egs in sd_loader:
        batchnum+=1
    print(batchnum)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to do speech separation in time domain using ConvTasNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="config file path")
    parser.add_argument(
        "--checkpoint", type=str, default="chkpt/xmix/x8515_xloss_ft/", help="Directory to dump models")
    parser.add_argument(
        "--metadata_dir", type=str, default="../LibriCse/Cse2Mix/60/metadata/mixture_train"
                                            "-100_mix_clean.csv", help="Data folder for development data.")
    parser.add_argument(
        "--sd_strategy", type=str, default=None, help="Directory to dump models")
    parser.add_argument(
        "--rttm_dir", type=str, default=None, help="Directory to dump models")
    parser.add_argument(
        "--log_name", type=str, default="test_cse.log", help="")
    parser.add_argument(
        "--description", type=str, default="0", help="")

    args = parser.parse_args()
    hp = Hparam(args.config)
    # diarization_precision(args, hp)
    run(args, hp)
