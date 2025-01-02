# !/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : Deep-Learning-Project-Template
@Product : PyCharm
@File : logger.py
@Author : Xuepu Zeng (2307665474zxp@gmail.com)
@Date : 2024/12/24 16:46
'''
import os
import sys
import functools
import logging
from typing import Optional
import torch.distributed as dist

def is_enabled() -> bool:
    """
    Returns:
        True if distributed training is enabled
    """
    return dist.is_available() and dist.is_initialized()

def get_global_rank() -> int:
    """
    Returns:
        The rank of the current process within the global process group.
    """
    return dist.get_rank() if is_enabled() else 0

def is_main_process() -> bool:
    """
    Returns:
        True if the current process is the main one.
    """
    return get_global_rank() == 0

# So that calling _configure_logger multiple times won't add many handlers
@functools.lru_cache()
def _configure_logger(
    name: Optional[str] = None,
    *,
    level: int = logging.DEBUG,
    output: Optional[str] = None,
    time_string: Optional[str] = None,
):
    """
    Configure a logger. Adapted from Detectron2.


    :param name: The name of the logger to configure.
    :param level: The logging level to use.
    :param output: A file name or a directory to save log. If None, will not save log file.
        If ends with ".txt" or ".log", assumed to be a file name.
        Otherwise, logs will be saved to `output/log.txt`.

    Returns: The configured logger.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Loosely match Google glog format:
    #   [IWEF]yyyymmdd hh:mm:ss.uuuuuu threadid file:line] msg
    # but use a shorter timestamp and include the logger name:
    #   [IWEF]yyyymmdd hh:mm:ss logger threadid file:line] msg
    fmt_prefix = "%(levelname).1s%(asctime)s %(name)s %(filename)s:%(lineno)s] "
    fmt_message = "%(message)s"
    fmt = fmt_prefix + fmt_message
    date_fmt = "%Y%m%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)

    # stdout logging for main worker only
    if is_main_process():
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # file logging for all workers
    if output:
        if os.path.splitext(output)[-1] in (".txt", ".log"):
            filename = output
        else:
            if time_string is None:
                filename = os.path.join(output, "logs", "log.txt")
            else:
                filename = os.path.join(output, "logs", f"log_{time_string}.txt")

        if not is_main_process():
            global_rank = get_global_rank()
            filename = filename + ".rank{}".format(global_rank)

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        handler = logging.StreamHandler(open(filename, "a"))
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

def init_logger(
        output: Optional[str] = None,
        name: Optional[str] = None,
        level: int = logging.DEBUG,
        capture_warnings: bool = True,
        time_string: Optional[str] = None) -> None:
    """
    Setup logging.

    Args:
        :param output: A file name or a directory to save log files. If None, log
            files will not be saved. If output ends with ".txt" or ".log", it
            is assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        :param name: The name of the logger to configure, by default the root logger.
        :param level: The logging level to use.
        :param capture_warnings: Whether warnings should be captured as logs.
        :param time_string:
    """
    logging.captureWarnings(capture_warnings)
    _configure_logger(name, level=level, output=output, time_string=time_string)