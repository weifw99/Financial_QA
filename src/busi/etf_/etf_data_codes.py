from src.busi.etf_.etf_data import EtfDataHandle


def main():
    code_list: list = ['SZ511880', 'SH159919', 'SZ510050', 'SZ510880']
    EtfDataHandle().get_down_data(code_list=code_list, refresh=True)


if __name__ == "__main__":
    main()

