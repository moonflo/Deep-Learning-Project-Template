# !/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : Deep-Learning-Project-Template
@Product : PyCharm
@File : data_spliter.py.py
@Author : Xuepu Zeng (2307665474zxp@gmail.com)
@Date : 2025/1/2 13:11
'''
import os
from typing import List

import pandas
import logging
import pandas as pd
from dataclasses import dataclass
from utils.misc import *

logger = logging.getLogger(__name__)


"""
Data format is demonstrated as below:
* fault name lookup table
+--------+------------+
| Number | Fault Name |
+--------+------------+
| 1      | str        |
+--------+------------+

* fault time lookup table
+---------+---------------+-------------------------+-------------------------+---------------+------------+---------------------+
| vin_md5 | Some Filed... | Start_Charging_Time(s)  | End_Charging_Time(s)    | Some Filed... | Fault_Name | Fault_Time          |
+---------+---------------+-------------------------+-------------------------+---------------+------------+---------------------+
| 0539d.. | Some Filed... | 2023-08-25 20:30:50.669 | 2023-08-25 21:20:18.931 | Some Filed... | 1          | 2023-09-14 19:43:34 |
+---------+---------------+-------------------------+-------------------------+---------------+------------+---------------------+

* raw data
+---------+-------------------------+---------------+
| vin_md5 | time                    | Some Filed... |
+---------+-------------------------+---------------+
| 0539d.. | 2023-06-27 18:23:14.538 | Some Filed... |
+---------+-------------------------+---------------+

"""
@dataclass
class Fault24121225:
    """
    MUST REMEMBER THIS DATA CANNOT BE UPLOADED TO PUBLIC
    """
    fault_name_LUT_path:    str="data\\fault_name_lut.xlsx"
    fault_time_LUT_path:    str="data\\fault_time_lut.xlsx"
    raw_data_path:          str="data\\raw_data.csv"

    def __post_init__(self):
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(self.raw_data_path)
        if not os.path.exists(self.fault_time_LUT_path):
            raise FileNotFoundError(self.fault_time_LUT_path)
        if not os.path.exists(self.fault_name_LUT_path):
            raise FileNotFoundError(self.fault_name_LUT_path)


@dataclass
class GroupRule:
    rules: List[str]

    def add_rules(self, _rules):
        if not isinstance(_rules, list):
            _rules = [_rules]
        self.rules.extend(_rules)

    def __str__(self):
        return_string = ""
        for rule in self.rules:
            return_string += rule + "\n"
        return return_string


class DataSplitter:
    def __init__(self, loading_cfg: Fault24121225, output_path: str, group_rule: GroupRule):
        self.raw_data_path = loading_cfg.raw_data_path
        self.fault_time_LUT_path = loading_cfg.fault_time_LUT_path
        self.fault_name_LUT_path = loading_cfg.fault_name_LUT_path
        self.out_put_path = output_path
        self.group_rule = group_rule

        self.raw_data_df = None
        self.fault_time_lut_df = None
        self.fault_name_lut_df = None

    def __load_file(self):
        logger.info("[INFO] Reading raw_data from {}...".format(self.raw_data_path))
        self.raw_data_df = safe_table_reader(self.raw_data_path)

        logger.info("[INFO] Reading Time LUT from {}...".format(self.fault_time_LUT_path))
        self.fault_time_lut_df = safe_table_reader(self.fault_time_LUT_path)

        logger.info("[INFO] Reading Name LUT from {}...".format(self.fault_name_LUT_path))
        self.fault_name_lut_df = safe_table_reader(self.fault_name_LUT_path)

        logger.info("[INFO] Data loading successfully.")

    def data_split(self):
        # No.1 Get time ranges
        if self.raw_data_df is None or self.fault_time_lut_df is None or self.fault_name_lut_df is None:
            self.__load_file()

        logger.info("[INFO] Starting group the raw data...")
        grouped_raw = self.fault_time_lut_df.groupby(self.group_rule.rules)
        time_ranges = grouped_raw.agg(
            start_time=('Start_Charging_Time(s)', 'min'),
            end_time=('End_Charging_Time(s)', 'max'))
        time_ranges = time_ranges.reset_index()

        # No.2 Using time range to chunk the raw
        raw_column_names = self.raw_data_df.columns.tolist()

        for _, row in time_ranges.iterrows():
            vin_md5 = row['vin_md5']
            start_charging_time = row['start_time']
            end_charging_time = row['end_time']
            fault_name = row['Fault_Name']

            subset = self.raw_data_df[
                (self.raw_data_df['vin_md5'] == vin_md5) &
                (self.raw_data_df['time'] >= start_charging_time) &
                (self.raw_data_df['time'] <= end_charging_time)
            ]

            if not subset.empty:
                subset.columns = raw_column_names
            else:
                raise "Cannot find data from timeranges: {} !".format(row)

            subset_save_path = os.path.join(self.out_put_path, "{}_{}.csv".format(vin_md5, fault_name))
            logger.info("[INFO] Saving data into: {}".format(subset_save_path))
            save_df(subset, subset_save_path)

        logger.info("[INFO] Data split successfully.")


if __name__ == '__main__':
    # 设置日志级别（可选，默认是 WARNING）
    logger.setLevel(logging.DEBUG)

    # 创建一个控制台处理器，输出到控制台
    console_handler = logging.StreamHandler()

    # 设置日志输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # 将处理器添加到logger中
    logger.addHandler(console_handler)


    fault24121225 = Fault24121225()
    group_rule = GroupRule(["vin_md5", "Fault_Name"])
    output_path = ".\\data\\Fault_data"

    data_split = DataSplitter(loading_cfg=fault24121225, output_path=output_path, group_rule=group_rule)

    data_split.data_split()
