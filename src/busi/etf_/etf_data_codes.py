from src.busi.etf_.etf_data import EtfDataHandle


def main():
    # code_list: list = ['SZ510050', 'SH159919', 'SZ513030', 'SZ511880', 'SZ510880',
    #                    'SZ518880', 'SZ513100', 'SZ510300', 'SH159915', 'SZ513520', 'SH159985']
    code_list: list = ['SZ510050', 'SH159919', ]
    EtfDataHandle().get_down_data(code_list=code_list, refresh=True)


if __name__ == "__main__":
    main()

