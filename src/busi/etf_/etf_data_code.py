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
    code_list: list[str] = ['SZ511880', 'SH159919', 'SZ510050', 'SZ510880']
    EtfDataHandle().get_down_etf_by_code(code_list=code_list)


if __name__ == "__main__":
    main()

