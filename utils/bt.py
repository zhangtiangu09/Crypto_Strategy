# -*- coding: utf-8 -*-
# @Author: Sky Zhang
# @Date:   2018-09-28 18:03:34
# @Last Modified by:   Sky Zhang
# @Last Modified time: 2018-09-28 18:03:57
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Performance:
    def __init__(self, ret):
        # ret: np.ndarray, daily return series
        assert isinstance(ret, np.ndarray)
        self.ret = ret

    def _info_ratio(self, bench):
        assert isinstance(bench, int) or (len(self.ret) == len(bench))
        excess = self.ret - bench
        ir = excess.mean() / excess.std()
        annualized_ir = ir * np.sqrt(252)
        return annualized_ir

    def _sharpe(self, int):
        return self._info_ratio(int)

    def _max_drawdown(self):
        x = np.cumprod(self.ret + 1)
        current_max = x[0]
        max_diff = 0
        for i in x[1:]:
            if i > current_max:
                current_max = i
            else:
                if (current_max - i) / current_max > max_diff:
                    max_diff = (current_max - i) / current_max
        return max_diff

    def _average_daily_ret(self):
        return self.ret.mean()

    def _vol(self):
        return self.ret.std() / np.sqrt(len(self.ret))

    def _win_rate(self):
        win = (self.ret > 0).sum()
        return win / len(self.ret)

    def summary(self, int=0, bench=None):
        mean = self._average_daily_ret() * 252
        vol = self._vol() * np.sqrt(252)
        max_drawdown = self._max_drawdown()
        win_rate = self._win_rate()
        annualized_sharpe = self._sharpe(int)
        summ = {'expected annualized return': mean,
                'annualized volatility': vol,
                'maximum drawdown': max_drawdown,
                'win ratio': win_rate,
                'anualized sharpe': annualized_sharpe}
        if bench:
            summ['annualized_info_ratio'] = self._info_ratio(bench)
        table = pd.DataFrame(summ, index=['summary']).T
        return table

    def plot(self):
        plt.plot(np.cumprod(self.ret + 1))
        plt.show()


class PortRet:
    def __init__(self, price, sig_func):
        # price: np.ndarray, price table, T*n, T periods and n assets
        self.price = price
        self.sig_func = sig_func

    def _signal_mat(self):
        def f(x):
            return list(self.sig_func(x))
        signal = [f[self.price[i, :]] for i in range(self.price.shape[0] - 1)]
        return np.array(signal)

    def portfolio_return(self):
        signal_mat = self._signal_mat()
        ret_mat = self.price[1:, :] / self.price[:-1, :] - 1
        return signal_mat @ ret_mat.T


def backtest(price, sig_func):
    PR = PortRet(price, sig_func)
    port_ret = PR.portfolio_return()
    PF = Performance(port_ret)
    return PF


def test1():
    ret = np.random.randn(10000) / np.sqrt(10000)
    performance = Performance(ret)
    print(np.cumsum(ret))
    print(performance.summary())
    performance.plot()


if __name__ == '__main__':
    test1()
