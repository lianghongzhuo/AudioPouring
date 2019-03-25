#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Hongzhuo Liang 
# E-mail     : liang@informatik.uni-hamburg.de
# Description: 
# Date       : 15/03/2019: 11:03 AM
# File Name  : generate_bottle_config
from __future__ import print_function, division
import numpy as np
import os
from audio_pouring.utils.utils import get_pouring_path
from numpy import genfromtxt
import matplotlib.pyplot as plt
import yaml

pkg_path = get_pouring_path()
config_file = os.path.join(pkg_path, "config/preprocess.yaml")
config = yaml.load(open(config_file, "r"))


def polyfit_with_fixed_points(degree_, x, y, x_fix, y_fix):
    """
    :param degree_: the degree of the polynomial
    :param x: input x
    :param y: input y
    :param x_fix: input fixed x
    :param y_fix: input fixed y
    :return:
    """
    mat = np.empty((degree_ + 1 + len(x_fix),) * 2)
    vec = np.empty((degree_ + 1 + len(x_fix),))
    x_n = x ** np.arange(2 * degree_ + 1)[:, None]
    yx_n = np.sum(x_n[:degree_ + 1] * y, axis=1)
    x_n = np.sum(x_n, axis=1)
    idx = np.arange(degree_ + 1) + np.arange(degree_ + 1)[:, None]
    mat[:degree_ + 1, :degree_ + 1] = np.take(x_n, idx)
    xf_n = x_fix ** np.arange(degree_ + 1)[:, None]
    mat[:degree_ + 1, degree_ + 1:] = xf_n / 2
    mat[degree_ + 1:, :degree_ + 1] = xf_n.T
    mat[degree_ + 1:, degree_ + 1:] = 0
    vec[:degree_ + 1] = yx_n
    vec[degree_ + 1:] = y_fix
    params = np.linalg.solve(mat, vec)
    return params[:degree_ + 1]


# fitting bottle 1, 3, 4, 6, 7, 8 height
def bottle_config(update_npy=True, vis=True):
    bottle_id_list = config["bottle_id_list"]
    for bottle_id in bottle_id_list:
        bottle_raw_data_path = os.path.join(pkg_path, "config/bottles", "bottle" + str(bottle_id) + "_config.csv")
        input_bottle_data = genfromtxt(bottle_raw_data_path, delimiter=",", skip_header=1)
        x = input_bottle_data[:, 1] / 1000.0
        y = input_bottle_data[:, 0]
        xf = np.array([input_bottle_data[0, 1] / 1000.0, input_bottle_data[-1, 1] / 1000.0])
        yf = np.array([input_bottle_data[0, 0], input_bottle_data[-1, 0]])
        params = polyfit_with_fixed_points(degree_=2, x=x, y=y, x_fix=xf, y_fix=yf)
        print("Bottle {} params are: {}".format(str(bottle_id), params))

        poly = np.polynomial.Polynomial(params)
        draw_line = np.linspace(x[0], x[-1], 50)

        if vis:
            plt.plot(x, y, "bo")
            plt.plot(xf, yf, "ro")
            plt.plot(draw_line, poly(draw_line), "-")
            plt.title("Bottle {} raw data fitting".format(str(bottle_id)))
            plt.xlabel("weight (g)")
            plt.ylabel("cavity height (mm)")
            plt.show()
        if update_npy:
            np.save(bottle_raw_data_path[:-4] + ".npy", params)


if __name__ == "__main__":
    bottle_config()
