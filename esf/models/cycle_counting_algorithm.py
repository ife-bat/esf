#!/usr/bin/env python
"""
This module uses the peak detection and rainflow libraries.
It enables to convert a random signal into simple cycles,
characterised by their range, count (half or full cycle)
and mean value.
"""


import numpy as np
import pandas as pd

import esf.external.peak_det.peak_det as pkd
import esf.external.rainflow_adapter as rf

# TODO: should be taken from a common "parameter or settings" module:
col_soc = "soc"
col_t = "t"
col_temperature = "temperature"
col_c_rate = "crate"


class CycleCounter:
    def __init__(
        self,
        time_v=None,
        soc_v=None,
        temp_v=None,
        crate_v=None,
        delta=0.1,
        verbose=False,
        **kwargs,
    ):

        assert (time_v is not None) and (
            soc_v is not None
        ), "Either a path or vectors must be argument of CycleCounter"

        t = time_v
        soc = soc_v

        self.mean_soc = 0
        self.arr_dod = []
        self.arr_n = []

        self.delta = delta

        self.data = pd.DataFrame(
            {col_t: t, col_soc: soc, col_temperature: temp_v, col_c_rate: crate_v}
        )
        self.min_points = []
        self.max_points = []

        self.arr_soc_mean = None
        self.arr_temp = None
        self.arr_time = None
        self.arr_crate = None

        self.verbose = verbose

        self.turning_points_extraction()

    def __str__(self):
        txt = "CycleCounter object\n"
        txt += f"mean_soc: {self.mean_soc}\n"
        txt += f"arr_dod: {self.arr_dod}\n"
        txt += f"arr_n: {self.arr_n}\n"
        txt += f"arr_soc_mean: {self.arr_soc_mean}\n"
        txt += f"arr_temp: {self.arr_temp}\n"
        txt += f"arr_time: {self.arr_time}\n"
        txt += f"arr_crate: {self.arr_crate}\n"
        return txt

    def turning_points_extraction(self):

        max_points, min_points = pkd.peakdet(self.data[col_soc], delta=self.delta)

        self.min_points = min_points
        self.max_points = max_points

    def rainflow_process_from_original_code(self):
        # check if arrays are empty, if so, adjust size to avoid concatenation issues:
        if self.min_points.size == 0:
            self.min_points = np.empty((0, 2))
        if self.max_points.size == 0:
            self.max_points = np.empty((0, 2))
        # concatenation of the turning points
        array_ext = np.concatenate((self.min_points, self.max_points), axis=0)
        array_ext = array_ext[array_ext[:, 0].argsort(), :]
        array_ext = np.transpose(array_ext)
        array_idx = [int(x) for x in array_ext[0]]
        array_ext = array_ext[1]

        # calculate cycle counts with the rainflow algorithm
        array_out = rf.rainflow(array_ext)

        # sort array_out by cycle range (lowest to highest dod)
        array_out = array_out[:, array_out[rf.ROW_RANGE, :].argsort()]

        self.mean_soc = np.mean(
            self.data[col_soc]
        )  # converts percentage into number between 0 and 1

        self.arr_dod = array_out[
            rf.ROW_RANGE, :
        ]  # difference between adjacent max/min points converted from % to number between 0 and 1
        self.arr_n = array_out[rf.ROW_COUNT, :]
        self.arr_soc_mean = array_out[
            rf.ROW_MEAN, :
        ]  # mean SOC in each interval converted from % to number between 0 and 1

        # get temperature, time, and c-rate
        temperature = self.data[col_temperature].values
        crate = self.data[col_c_rate].values
        time = self.data[col_t].values

        cycle_temp = []
        cycle_time = []
        cycle_crate = []
        idx_0 = 0
        for n in self.arr_n:
            idx_1 = int(idx_0 + n * 2)

            _temperature = np.mean(temperature[array_idx[idx_0] : array_idx[idx_1]])
            _c_rate = np.mean(crate[array_idx[idx_0] : array_idx[idx_1]])
            _delta_time = time[array_idx[idx_1]] - time[array_idx[idx_0]]

            cycle_temp.append(_temperature)
            cycle_crate.append(_c_rate)
            cycle_time.append(_delta_time)

            idx_0 = idx_1

        self.arr_temp = np.array(cycle_temp)
        self.arr_time = np.array(cycle_time)
        self.arr_crate = np.array(cycle_crate)

    def rainflow_process(self):
        # from the modified code (Jinsung?)
        if len(self.min_points) > 0 and len(self.max_points) > 0:
            # we have both peaks and valleys

            # concatenation of the turning points
            array_ext = np.concatenate((self.min_points, self.max_points), axis=0)
            array_ext = array_ext[array_ext[:, 0].argsort(), :]
            array_ext = np.transpose(array_ext)
            array_idx = [int(x) for x in array_ext[0]]
            array_ext = array_ext[1]

            # calculate cycle counts with the rainflow algorithm
            array_out = rf.rainflow(array_ext)

            self.mean_soc = np.mean(self.data[col_soc])
            self.arr_dod = array_out[rf.ROW_RANGE, :]
            self.arr_soc_mean = array_out[rf.ROW_MEAN, :]
            self.arr_n = array_out[rf.ROW_COUNT, :]

            # Get cycle temperature
            temperature = self.data[col_temperature].values
            crate = self.data[col_c_rate].values
            time = self.data[col_t].values

            cycle_temp = []
            cycle_time = []
            cycle_crate = []
            idx_0 = 0
            for n in self.arr_n:
                idx_1 = int(idx_0 + n * 2)

                _temperature = np.mean(temperature[array_idx[idx_0] : array_idx[idx_1]])
                _c_rate = np.mean(crate[array_idx[idx_0] : array_idx[idx_1]])
                _delta_time = time[array_idx[idx_1]] - time[array_idx[idx_0]]

                cycle_temp.append(_temperature)
                cycle_crate.append(_c_rate)
                cycle_time.append(_delta_time)

                idx_0 = idx_1

            self.arr_temp = np.array(cycle_temp)
            self.arr_time = np.array(cycle_time)
            self.arr_crate = np.array(cycle_crate)

        else:
            self.mean_soc = np.mean(self.data[col_soc])
            self.arr_dod = [0.0]
            self.arr_n = [1]
            self.arr_soc_mean = [self.mean_soc]

            self.arr_temp = [np.mean(self.data[col_temperature])]

            _time = self.data[col_t].values
            self.arr_time = [_time[-1] - _time[0]]
            self.arr_crate = [np.mean(self.data[col_c_rate])]
