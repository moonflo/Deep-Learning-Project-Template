# !/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : Deep-Learning-Project-Template
@Product : PyCharm
@File : sort_rule.py.py
@Author : Xuepu Zeng (2307665474zxp@gmail.com)
@Date : 2025/1/2 13:19
'''

from dataclasses import dataclass, field
from typing import List, Callable
import operator

# Sort rule class
@dataclass
class SortRule:
    field: str              # Field name for sorting
    ascending: bool = True  # If ascending

    def __post_init__(self):
        # if descending, change the sorting func
        self.sort_func = operator.lt if not self.ascending else operator.gt