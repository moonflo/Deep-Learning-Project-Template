# !/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : Deep-Learning-Project-Template
@Product : PyCharm
@File : misc.py.py
@Author : Xuepu Zeng (2307665474zxp@gmail.com)
@Date : 2025/1/2 14:23
'''
import os
import pandas as pd
import logging
logger = logging.getLogger()


def safe_table_reader(read_file_path: str) -> pd.DataFrame:
    if not os.path.exists(read_file_path):
        raise FileNotFoundError(read_file_path)
    elif read_file_path.endswith('.csv'):
        return pd.read_csv(read_file_path)
    elif read_file_path.endswith('.xlsx'):
        return pd.read_excel(read_file_path)
    else:
        raise "File type not supported!"


def save_df(df: pd.DataFrame, file_path: str):
    df.to_csv(file_path, index=False)

