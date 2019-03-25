#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Hongzhuo Liang 
# E-mail     : liang@informatik.uni-hamburg.de
# Description: move 20% of file from train to test
# Date       : 20/03/2019: 1:30 PM
# File Name  : spilt_train_test_data

import glob
import os
import shutil
if __name__ == "__main__":
    for bottle in [1, 3, 4]:
        pickle = glob.glob(os.path.join("./pickle_train_" + str(bottle), "*.pickle"))
        pickle_test = glob.glob(os.path.join("./pickle_test_" + str(bottle), "*.pickle"))
        pickle_num = len(pickle)
        pickle_test_num = len(pickle_test)
        if pickle_test_num == 0:
            if pickle_num != 0:
                train_num = int(pickle_num * 0.8)
                test_num = pickle_num - train_num
                for test_file in pickle[:test_num]:
                    shutil.move(test_file, "pickle_test_" + str(bottle) + "/.")
                print("Done, bottle {}".format(bottle))
            else:
                print("Pickle train folder for bottle {} is empty".format(bottle))
        else:
            print("Pickle test folder for bottle {} is not empty".format(bottle))
