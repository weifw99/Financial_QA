import backtrader as bt
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split


class LightGBMFactorStrategy(bt.Strategy):
    params = (
        ('rebalance_freq', 21),  # 月频再平衡，约21个交易日
        ('prediction_horizon', 20),  # 预测20天后的上涨
        ('volatility_target', 0.07),  # 目标年化波动率7%
        ('volatility_lookback', 20),  # 波动率计算窗口
        ('train_lookback', 252 * 3),  # 训练数据回溯3年
        ('min_probability', 0.5),  # 上涨概率阈值
        ('position_sizing', 0.25),  # 单个资产基础仓位
    )

    def __init__(self):
        # 数据引用
        self.stock = self.datas[0]  # 沪深300 ETF
        self.gold = self.datas[1]  # 黄金 ETF
        self.bond = self.datas[2]  # 国债 ETF
        self.money = self.datas[3]  # 银华日利

        # 训练模型用
        self.models = {
            'stock': None,
            'gold': None
        }

        # 初始化计数器
        self.days_passed = 0
        self.last_rebalance = 0

        # 跟踪仓位
        self.positions_info = {
            'stock': 0,
            'gold': 0,
            'bond': 0,
            'money': 0
        }

        # 存储历史数据用于因子计算
        self.hist_data = {
            'stock': [],
            'gold': [],
            'bond': [],
            'money': []
        }

    def calculate_technical_factors(self, data_array):
        """计算技术因子"""
        if len(data_array) < 50:
            return None

        df = pd.DataFrame(data_array, columns=['close', 'volume', 'high', 'low', 'open'])

        # 价格动量因子
        df['returns_5'] = df['close'].pct_change(5)
        df['returns_10'] = df['close'].pct_change(10)
        df['returns_20'] = df['close'].pct_change(20)

        # 移动平均线
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()

        # 价格突破
        df['price_vs_ma5'] = df['close'] / df['ma5'] - 1
        df['price_vs_ma20'] = df['close'] / df['ma20'] - 1

        # 波动率因子
        df['volatility_20'] = df['returns_20'].rolling(20).std()

        # 成交量因子
        df['volume_ma5'] = df['volume'].rolling(5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma5']

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # 价格通道
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20'] = df['low'].rolling(20).min()
        df['atr'] = self.calculate_atr(df)

        # 清理NaN值
        df = df.dropna()

        return df

    def calculate_atr(self, df, period=14):
        """计算ATR"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()
        return atr

    def prepare_training_data(self, asset_name, data_array):
        """准备训练数据"""
        df = self.calculate_technical_factors(data_array)
        if df is None or len(df) < 100:
            return None, None

        # 创建标签：20天后是否上涨
        df['future_return'] = df['close'].shift(-self.params.prediction_horizon) / df['close'] - 1
        df['label'] = (df['future_return'] > 0).astype(int)

        # 选择特征列
        feature_cols = ['returns_5', 'returns_10', 'returns_20',
                        'price_vs_ma5', 'price_vs_ma20',
                        'volatility_20', 'volume_ratio', 'rsi',
                        'high_20', 'low_20', 'atr']

        features = df[feature_cols].iloc[:-self.params.prediction_horizon]
        labels = df['label'].iloc[:-self.params.prediction_horizon]

        return features, labels

    def train_lightgbm_model(self, features, labels):
        """训练LightGBM模型"""
        if len(features) < 100:
            return None

        # 分割训练验证集
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )

        # 创建数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # 设置参数
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'min_data_in_leaf': 20
        }

        print('--------------------------------训练模型')
        # 训练模型
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=100,
            callbacks=[lgb.early_stopping(10)]
        )

        return model

    def calculate_volatility(self, prices, lookback=20):
        """计算年化波动率"""
        if len(prices) < lookback:
            return 0

        returns = np.diff(np.log(prices[-lookback:]))
        daily_vol = np.std(returns)
        annual_vol = daily_vol * np.sqrt(252)
        return annual_vol

    def get_current_features(self, asset_name):
        """获取当前时刻的特征"""
        if len(self.hist_data[asset_name]) < 50:
            return None

        recent_data = self.hist_data[asset_name][-50:]
        df = self.calculate_technical_factors(recent_data)

        if df is None or len(df) == 0:
            return None

        # 选择特征列
        feature_cols = ['returns_5', 'returns_10', 'returns_20',
                        'price_vs_ma5', 'price_vs_ma20',
                        'volatility_20', 'volume_ratio', 'rsi',
                        'high_20', 'low_20', 'atr']

        # 获取最新特征
        current_features = df[feature_cols].iloc[-1:]
        return current_features

    def predict_probability(self, asset_name):
        """预测上涨概率"""
        model = self.models[asset_name]
        if model is None:
            return 0.5  # 默认值

        features = self.get_current_features(asset_name)
        if features is None:
            return 0.5

        try:
            prob = model.predict(features)[0]
            return float(prob)
        except:
            return 0.5

    def calculate_target_weights(self):
        """计算目标权重"""
        # 获取当前波动率
        stock_prices = [d[0] for d in self.hist_data['stock']]
        gold_prices = [d[0] for d in self.hist_data['gold']]

        stock_vol = self.calculate_volatility(stock_prices, self.params.volatility_lookback)
        gold_vol = self.calculate_volatility(gold_prices, self.params.volatility_lookback)

        # 使用平均波动率作为组合波动率估计
        portfolio_vol = (stock_vol + gold_vol) / 2 if stock_vol > 0 and gold_vol > 0 else 0

        # 波动率调整因子
        if portfolio_vol > 0:
            vol_adjustment = min(1.0, self.params.volatility_target / portfolio_vol)
        else:
            vol_adjustment = 1.0

        # 基础风险仓位
        base_risk_weight = self.params.position_sizing * 2  # 股票+黄金

        # 波动率调整后的风险仓位
        adjusted_risk_weight = base_risk_weight * vol_adjustment

        # 获取预测概率
        stock_prob = self.predict_probability('stock')
        gold_prob = self.predict_probability('gold')

        # 根据预测调整仓位
        if stock_prob < self.params.min_probability:
            stock_weight = 0
        else:
            stock_weight = adjusted_risk_weight * (stock_prob - 0.5) * 2

        if gold_prob < self.params.min_probability:
            gold_weight = 0
        else:
            gold_weight = adjusted_risk_weight * (gold_prob - 0.5) * 2

        # 确保权重非负且不超过上限
        stock_weight = max(0, min(0.5, stock_weight))
        gold_weight = max(0, min(0.5, gold_weight))

        # 风险资产总权重
        risk_weight = stock_weight + gold_weight

        # 剩余权重分配给债券和货币基金
        remaining_weight = 1 - risk_weight

        # 债券和货币基金平分剩余权重
        bond_weight = remaining_weight * 0.5
        money_weight = remaining_weight * 0.5

        # 如果股票和黄金仓位都为0，增加债券和货币基金仓位
        if risk_weight == 0:
            bond_weight = 0.5
            money_weight = 0.5

        return {
            'stock': stock_weight,
            'gold': gold_weight,
            'bond': bond_weight,
            'money': money_weight
        }

    def next(self):
        """每天执行"""
        # 记录历史数据
        self.hist_data['stock'].append([
            self.stock.close[0],
            self.stock.volume[0],
            self.stock.high[0],
            self.stock.low[0],
            self.stock.open[0]
        ])

        self.hist_data['gold'].append([
            self.gold.close[0],
            self.gold.volume[0],
            self.gold.high[0],
            self.gold.low[0],
            self.gold.open[0]
        ])

        self.hist_data['bond'].append([
            self.bond.close[0],
            self.bond.volume[0],
            self.bond.high[0],
            self.bond.low[0],
            self.bond.open[0]
        ])

        self.hist_data['money'].append([
            self.money.close[0],
            self.money.volume[0],
            self.money.high[0],
            self.money.low[0],
            self.money.open[0]
        ])

        self.days_passed += 1

        # 检查是否需要再平衡
        if self.days_passed - self.last_rebalance >= self.params.rebalance_freq:
            self.rebalance_portfolio()
            self.last_rebalance = self.days_passed

    def rebalance_portfolio(self):
        """执行再平衡"""
        # 训练模型（每3个月训练一次）
        if self.days_passed % (63) == 0 and len(self.hist_data['stock']) > self.params.train_lookback:
            # 训练股票模型
            stock_data = self.hist_data['stock'][-self.params.train_lookback:]
            stock_features, stock_labels = self.prepare_training_data('stock', stock_data)
            if stock_features is not None:
                self.models['stock'] = self.train_lightgbm_model(stock_features, stock_labels)

            # 训练黄金模型
            gold_data = self.hist_data['gold'][-self.params.train_lookback:]
            gold_features, gold_labels = self.prepare_training_data('gold', gold_data)
            if gold_features is not None:
                self.models['gold'] = self.train_lightgbm_model(gold_features, gold_labels)

        # 计算目标权重
        target_weights = self.calculate_target_weights()
        print(target_weights)

        # 执行调仓
        portfolio_value = self.broker.getvalue()

        # 计算目标市值
        target_values = {
            'stock': portfolio_value * target_weights['stock'],
            'gold': portfolio_value * target_weights['gold'],
            'bond': portfolio_value * target_weights['bond'],
            'money': portfolio_value * target_weights['money']
        }

        # 调整仓位
        assets = {
            'stock': self.stock,
            'gold': self.gold,
            'bond': self.bond,
            'money': self.money
        }

        for asset_name, data in assets.items():
            current_value = self.getposition(data).size * data.close[0]
            target_value = target_values[asset_name]

            if target_value > current_value:
                # 买入
                size = int((target_value - current_value) / data.close[0])
                if size > 0:
                    self.buy(data=data, size=size)
            elif target_value < current_value:
                # 卖出
                size = int((current_value - target_value) / data.close[0])
                if size > 0:
                    self.sell(data=data, size=size)

        # 记录仓位信息
        self.positions_info = target_weights

    def stop(self):
        """策略结束"""
        print("策略运行结束")
        print(f"最终资产组合价值: {self.broker.getvalue():.2f}")
        print(f"最终仓位配置:")
        print(f"  沪深300: {self.positions_info['stock'] * 100:.1f}%")
        print(f"  黄金: {self.positions_info['gold'] * 100:.1f}%")
        print(f"  国债: {self.positions_info['bond'] * 100:.1f}%")
        print(f"  银华日利: {self.positions_info['money'] * 100:.1f}%")


# 创建并运行回测
def run_backtest():
    cerebro = bt.Cerebro()

    # 设置初始资金
    cerebro.broker.setcash(1000000)

    # 设置佣金
    cerebro.broker.setcommission(commission=0.0003)  # 0.03%佣金

    from busi.etf_.bt_data import Getdata

    pool_file = 'data/etf_strategy/etf_pool_120.csv'
    pool_file = 'data/etf_strategy/etf_pool.csv'
    pool_file = 'data/etf_strategy/etf_pool1.csv'
    df = pd.read_csv(pool_file)
    etf_codes = df['代码'].tolist()
    etf_codes: list = ['SZ510300', 'SZ511880', 'SZ510880',
                       'SZ518880', 'SZ513100', 'SZ510300', 'SH159915', 'SZ513520', 'SH159985']
    # 获取数据源
    datas = Getdata(symbols=etf_codes)
    data_1 = datas.dailydata_no_index()

    # 示例：df = pd.read_csv('etf_data.csv', parse_dates=['date'])
    df = data_1.sort_values(['symbol', 'date']).copy()

    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(['symbol', 'date'], inplace=True)

    # 四个ETF代码
    etf_codes = [
        ('SZ510300', '沪深300ETF'),
        ('SZ511160', '国债ETF'), # 511260
        ('SZ518880', '黄金ETF'),
        ('SZ511880', '货币基金')
    ]

    # 加载数据
    for code, name in etf_codes:
        #try:
            # 从2015年开始，与论文一致
        df = loader.read_df([code], start_date='20170823')

        if df is None or df.empty:
            print(f"  警告: {code} 数据为空")
            continue

        # 准备数据
        df.index = pd.to_datetime(df['date'])

        # 确保必要的列存在
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                print(f"  警告: {code} 缺少{col}列")
                break
        else:
            data = bt.feeds.PandasData(
                dataname=df[required_cols],
                name=f"{code}({name})"
            )
            cerebro.adddata(data)
            print(f"  已加载: {name}")

    # 添加策略
    cerebro.addstrategy(LightGBMFactorStrategy)

    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='_PyFolio')
    #cerebro.addanalyzer(bt.analyzers.Volatility, _name='volatility')



    # 运行回测
    print('初始资金: %.2f' % cerebro.broker.getvalue())
    results = cerebro.run()
    print('最终资金: %.2f' % cerebro.broker.getvalue())

    # 输出分析结果
    strat = results[0]

    print("\n=== 回测结果 ===")
    print(f"年化收益率: {strat.analyzers.returns.get_analysis()['rnorm100']:.2f}%")
    print(f"夏普比率: {strat.analyzers.sharpe.get_analysis()['sharperatio']:.2f}")
    print(f"最大回撤: {strat.analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%")
    #print(f"年化波动率: {strat.analyzers.volatility.get_analysis()['volatility']:.2f}")


    portfolio_stats = strat.analyzers.getbyname('_PyFolio')
    returns, positions, transactions, _ = portfolio_stats.get_pf_items()
    returns.index = returns.index.tz_convert(None)

    cumulative_returns = (1 + returns).cumprod() - 1
    cumulative_returns.plot(title='Cumulative Returns')
    import matplotlib.pyplot as plt
    plt.show()

    # 绘制图表
    #cerebro.plot()


if __name__ == '__main__':
    run_backtest()