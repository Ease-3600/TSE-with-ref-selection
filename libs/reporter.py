#!/usr/bin/env python

import sys
sys.path.append("../")

from utils.timer import Timer
import torch


class Reporter(object):
    """
    A progress reporter to record the loss for each batch
    """

    def __init__(self, logger, period=100):
        self.period = period
        self.logger = logger
        self.loss = []
        self.ce = []
        self.ncorrect = []
        self.nsample = []
        self.mae = []
        self.sdr = []
        self.timer = Timer()

    def add(self, loss, ce, ncorrect, nsample, mae, sdr):
        self.loss.append(loss)
        self.ce.append(ce)
        self.ncorrect.append(ncorrect)
        self.nsample.append(nsample)
        self.mae.append(mae)
        self.sdr.append(sdr)
        N = len(self.loss)
        if not N % self.period:
            avg = sum(self.loss[-self.period:]) / self.period
            avg_ce = sum(self.ce[-self.period:]) / self.period
            acc = sum(self.ncorrect[-self.period:]) / sum(self.nsample[-self.period:]) * 100.0
            avg_mae = sum(self.mae[-self.period:]) / (self.period - self.mae[-self.period:].count(torch.tensor(0)))
            avg_sdr = sum(self.sdr[-self.period:]) / (self.period - self.sdr[-self.period:].count(torch.tensor(0)))
            self.logger.info("Processed {:d} batches"
                    "(loss = {:+.2f}, ce = {:f}, accuracy = {:+.2f}), "
                    "absent_mae = {:+.2f}, present_sdr = {:+.2f}...".format(N, avg, avg_ce, acc, avg_mae, avg_sdr))

    def report(self, details=False):
        N = len(self.loss)
        if details:
            sstr = ",".join(map(lambda f: "{:.2f}".format(f), self.loss))
            self.logger.info("Loss on {:d} batches: {}".format(N, sstr))
            sstr = ",".join(map(lambda f: "{:.4f}".format(f), self.ce))
            self.logger.info("Loss on {:d} batches: {}".format(N, sstr))
        return {
            "loss": sum(self.loss) / N,
            "ce": sum(self.ce) / N,
            "accuracy": sum(self.ncorrect) / sum(self.nsample) * 100.0,
            "batches": N,
            "cost": self.timer.elapsed(),
            "mae": sum(self.mae) / (N - self.mae.count(torch.tensor(0))),
            "sdr": sum(self.sdr) / (N - self.sdr.count(torch.tensor(0)))
        }
