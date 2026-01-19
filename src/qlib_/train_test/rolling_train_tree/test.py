import qlib
from qlib.data import D

qlib.init(provider_uri="/Users/dabai/liepin/study/llm/Financial_QA/data/qlib_data/cn_data_etf")

instruments = D.instruments(market='all')
print(instruments )
print(D.list_instruments(instruments=["SZ513520"]))