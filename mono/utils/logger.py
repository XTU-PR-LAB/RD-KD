import atexit
import logging
import os
import sys
import time
import torch
from termcolor import colored

__all__ = ["setup_logger", ]

class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log

def setup_logger(
    output=None, distributed_rank=0, *, name='metricdepth', color=True, abbrev_name=None
):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if abbrev_name is None:
        abbrev_name = "d2"
        
    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s %(message)s ", datefmt="%m/%d %H:%M:%S"
    )
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
                abbrev_name=str(abbrev_name),
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if distributed_rank > 0:
            filename = filename + ".rank{}".format(distributed_rank)
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.INFO)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)
    

    return logger

from iopath.common.file_io import PathManager as PathManagerBase


PathManager = PathManagerBase()

def _cached_log_stream(filename):
    io = PathManager.open(filename, "a", buffering=1024 if "://" in filename else -1)
    atexit.register(io.close)
    return io
    