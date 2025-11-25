import random
import traceback
from typing import Union, List, Optional, Tuple

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path
import glob
import json
import time

from scipy.stats import linregress

import sys

from src.busi.etf_.etf_data import EtfDataHandle
from src.busi.etf_.constant import DataCons


def main():
    # EtfDataHandle().get_etf_data()
    # code_list: list = ['SZ510050', 'SH159919', 'SZ513030', 'SZ511880', 'SZ510880',
    #                    'SZ518880', 'SZ513100', 'SZ510300', 'SH159915', 'SZ513520', 'SH159985']
    code_list: list = ['SZ510050', 'SH159919', ]
    EtfDataHandle().get_down_etf_by_code(code_list=code_list)


if __name__ == "__main__":
    main()

