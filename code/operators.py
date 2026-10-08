# -*- coding: utf-8 -*-
"""
Alpha191 时序/截面算子库
所有算子作用于宽表（行=交易日，列=股票代码，pd.DataFrame），
时序算子按列（个股）计算，RANK 为截面算子（按行）。
公式与函数定义来源：国泰君安《基于短周期价量特征的多因子选股体系》(2017-06-15) 表6及附录2。
"""
import numpy as np
import pandas as pd
import bottleneck as bn


# ---------- 基础时序算子 ----------
def DELAY(x, n):
    return x.shift(n)


def DELTA(x, n):
    return x - x.shift(n)


def SUM(x, n):
    return x.rolling(n, min_periods=n).sum()


def MEAN(x, n):
    return x.rolling(n, min_periods=n).mean()


def STD(x, n):
    return x.rolling(n, min_periods=n).std(ddof=1)


def TSMAX(x, n):
    return x.rolling(n, min_periods=n).max()


def TSMIN(x, n):
    return x.rolling(n, min_periods=n).min()


def CORR(a, b, n):
    return a.rolling(n, min_periods=n).corr(b)


def COVANCE(a, b, n):
    return a.rolling(n, min_periods=n).cov(b)


def CORR_FULL(a, b):
    """全历史（expanding）相关系数，用于 Alpha36 等无窗口 CORR"""
    return a.expanding(min_periods=10).corr(b)


def LOG(x):
    return np.log(x.where(x > 0))


def ABS(x):
    return x.abs()


def SIGN(x):
    return np.sign(x)


def POWER(a, b):
    """标准数学幂运算：负数底数的整数次幂正常计算，非整数次幂为 NaN"""
    with np.errstate(invalid='ignore'):
        return np.power(a.astype(float), b.astype(float) if hasattr(b, 'astype') else b)


# ---------- 截面算子 ----------
def RANK(x):
    """截面升序秩，归一化到 (0,1]；附录2: RANK(A)=向量A升序排序"""
    return x.rank(axis=1, pct=True)


# ---------- 高级算子 ----------
def SMA(x, n, m):
    """通达信口径 SMA: Y=(X*m+Y'*(n-m))/n == ewm(alpha=m/n, adjust=False)"""
    return x.ewm(alpha=m / n, adjust=False, min_periods=1).mean()


SMEAN = SMA  # Alpha22 中的 SMEAN 按 SMA 处理（公开实现惯例）


def WMA(x, n):
    """加权平均，权重 0.9^i（i=距当前时点的间隔），归一化"""
    w = 0.9 ** np.arange(n - 1, -1, -1)  # 当前点权重 0.9^0=1
    w = w / w.sum()
    return x.rolling(n, min_periods=n).apply(lambda v: np.dot(v, w), raw=True)


def DECAYLINEAR(x, n):
    """移动加权平均，权重 n, n-1, ..., 1，权重和为1"""
    w = np.arange(n, 0, -1, dtype=float)
    w = w / w.sum()
    return x.rolling(n, min_periods=n).apply(lambda v: np.dot(v, w), raw=True)


def TSRANK(x, n):
    """末位值在过去n天的顺序排位（1..n）。
    bottleneck.move_rank 输出范围 [-1,1]（-1=最小, +1=最大），映射回序数秩。"""
    vals = x.values.astype(float)
    ranked = bn.move_rank(vals, window=n, axis=0)          # [-1,1]
    out = (ranked + 1) / 2 * (n - 1) + 1                   # → [1,n]
    res = pd.DataFrame(out, index=x.index, columns=x.columns)
    cnt = x.notna().rolling(n, min_periods=n).count()
    res[cnt < n] = np.nan
    return res


def COUNT(cond, n):
    return cond.astype(float).rolling(n, min_periods=n).sum()


def SUMIF(x, n, cond):
    return x.where(cond, 0.0).rolling(n, min_periods=1).sum()


def FILTER(x, cond):
    return x.where(cond)


def PROD(x, n):
    return x.rolling(n, min_periods=n).apply(np.prod, raw=True)


def SUMAC(x, n=None):
    """累加（cumsum）；附录：SUMAC(A,n)=前n项累加，实际出现于全历史场景"""
    return x.cumsum()


def FMAX(x):
    """全历史最大值（标量按列广播），用于 Alpha165/183 中无窗口 MAX"""
    row = np.nanmax(x.values, axis=0)
    return pd.DataFrame(np.tile(row, (len(x), 1)), index=x.index, columns=x.columns)


def FMIN(x):
    row = np.nanmin(x.values, axis=0)
    return pd.DataFrame(np.tile(row, (len(x), 1)), index=x.index, columns=x.columns)


def HIGHDAY(x, n):
    """前n期最大值距当前时点的间隔"""
    am = bn.move_argmax(x.values, window=n, axis=0)
    out = (n - 1) - am
    res = pd.DataFrame(out, index=x.index, columns=x.columns).astype(float)
    res[x.rolling(n, min_periods=n).count().isna()] = np.nan
    return res


def LOWDAY(x, n):
    """前n期最小值距当前时点的间隔"""
    am = bn.move_argmin(x.values, window=n, axis=0)
    out = (n - 1) - am
    res = pd.DataFrame(out, index=x.index, columns=x.columns).astype(float)
    res[x.rolling(n, min_periods=n).count().isna()] = np.nan
    return res


def REGBETA(a, b, n):
    """前n期 a 对 b 回归的回归系数（斜率）"""
    cov = a.rolling(n, min_periods=n).cov(b)
    var = b.rolling(n, min_periods=n).var(ddof=1)
    return cov / var.replace(0, np.nan)


def REGRESI(a, b, n):
    """前n期 a 对 b 回归、最后一个样本点的残差"""
    beta = REGBETA(a, b, n)
    ma = a.rolling(n, min_periods=n).mean()
    mb = b.rolling(n, min_periods=n).mean()
    alpha = ma - beta * mb
    return a - (alpha + beta * b)


def SEQUENCE(n, index, columns):
    """生成 1~n 等差序列并平铺成宽表（用于 REGBETA(y, SEQUENCE(n))）"""
    v = np.tile(np.arange(1, n + 1, dtype=float), (len(index), 1))
    df = pd.DataFrame(v, index=index, columns=['__seq__'])
    return df
