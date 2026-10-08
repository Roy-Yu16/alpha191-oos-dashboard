# -*- coding: utf-8 -*-
"""Alpha191 因子实现（Alpha101 ~ Alpha150）"""
import numpy as np
import pandas as pd
from operators import (DELAY, DELTA, SUM, MEAN, STD, TSMAX, TSMIN, CORR, COVANCE,
                       LOG, ABS, SIGN, POWER, RANK, SMA, WMA, DECAYLINEAR,
                       TSRANK, COUNT, SUMIF, FILTER, PROD, HIGHDAY, LOWDAY,
                       REGBETA, REGRESI)
from alphas_001_050 import _regbeta_seq
from alphas_051_100 import _rsi_like


def alpha101(ctx):
    return (RANK(CORR(ctx.C, SUM(MEAN(ctx.V, 30), 37), 15))
            < RANK(CORR(RANK(ctx.H * 0.1 + ctx.VWAP * 0.9), RANK(ctx.V), 11))
            ).astype(float) * -1


def alpha102(ctx):
    return _rsi_like(ctx.V, 6)


def alpha103(ctx):
    return (20 - LOWDAY(ctx.L, 20)) / 20 * 100


def alpha104(ctx):
    return -1 * DELTA(CORR(ctx.H, ctx.V, 5), 5) * RANK(STD(ctx.C, 20))


def alpha105(ctx):
    return -1 * CORR(RANK(ctx.O), RANK(ctx.V), 10)


def alpha106(ctx):
    return ctx.C - DELAY(ctx.C, 20)


def alpha107(ctx):
    return (-1 * RANK(ctx.O - DELAY(ctx.H, 1))) * RANK(ctx.O - DELAY(ctx.C, 1)) \
        * RANK(ctx.O - DELAY(ctx.L, 1))


def alpha108(ctx):
    return POWER(RANK(ctx.H - TSMIN(ctx.H, 2)),
                 RANK(CORR(ctx.VWAP, MEAN(ctx.V, 120), 6))) * -1


def alpha109(ctx):
    r = ctx.H - ctx.L
    return SMA(r, 10, 2) / SMA(SMA(r, 10, 2), 10, 2)


def alpha110(ctx):
    c1 = DELAY(ctx.C, 1)
    up = SUM(pd.DataFrame(np.maximum(0.0, ctx.H - c1), index=ctx.C.index, columns=ctx.C.columns), 20)
    dn = SUM(pd.DataFrame(np.maximum(0.0, c1 - ctx.L), index=ctx.C.index, columns=ctx.C.columns), 20)
    return up / dn.replace(0, np.nan) * 100


def alpha111(ctx):
    x = ctx.V * ((ctx.C - ctx.L) - (ctx.H - ctx.C)) / (ctx.H - ctx.L)
    return SMA(x, 11, 2) - SMA(x, 4, 2)


def alpha112(ctx):
    d = ctx.C - DELAY(ctx.C, 1)
    up = SUM(d.where(d > 0, 0.0), 12)
    dn = SUM(ABS(d.where(d < 0, 0.0)), 12)
    return (up - dn) / (up + dn).replace(0, np.nan) * 100


def alpha113(ctx):
    return -1 * (RANK(SUM(DELAY(ctx.C, 5), 20) / 20) * CORR(ctx.C, ctx.V, 2)
                 * RANK(CORR(SUM(ctx.C, 5), SUM(ctx.C, 20), 2)))


def alpha114(ctx):
    x = (ctx.H - ctx.L) / MEAN(ctx.C, 5)
    return (RANK(DELAY(x, 2)) * RANK(RANK(ctx.V))) / (x / (ctx.VWAP - ctx.C))


def alpha115(ctx):
    return POWER(RANK(CORR(ctx.H * 0.9 + ctx.C * 0.1, MEAN(ctx.V, 30), 10)),
                 RANK(CORR(TSRANK((ctx.H + ctx.L) / 2, 4), TSRANK(ctx.V, 10), 7)))


def alpha116(ctx):
    return _regbeta_seq(ctx.C, 20)  # 原文 REGBETA(CLOSE,SEQUENCE,20)，SEQUENCE 后应为窗口 20


def alpha117(ctx):
    return (TSRANK(ctx.V, 32) * (1 - TSRANK(ctx.C + ctx.H - ctx.L, 16))) \
        * (1 - TSRANK(ctx.RET, 32))


def alpha118(ctx):
    return SUM(ctx.H - ctx.O, 20) / SUM(ctx.O - ctx.L, 20).replace(0, np.nan) * 100


def alpha119(ctx):
    return (RANK(DECAYLINEAR(CORR(ctx.VWAP, SUM(MEAN(ctx.V, 5), 26), 5), 7))
            - RANK(DECAYLINEAR(TSRANK(TSMIN(CORR(RANK(ctx.O), RANK(MEAN(ctx.V, 15)), 21), 9), 7), 8)))


def alpha120(ctx):
    return RANK(ctx.VWAP - ctx.C) / RANK(ctx.VWAP + ctx.C)


def alpha121(ctx):
    return POWER(RANK(ctx.VWAP - TSMIN(ctx.VWAP, 12)),
                 TSRANK(CORR(TSRANK(ctx.VWAP, 20), TSRANK(MEAN(ctx.V, 60), 2), 18), 3)) * -1


def alpha122(ctx):
    s = SMA(SMA(SMA(LOG(ctx.C), 13, 2), 13, 2), 13, 2)
    return (s - DELAY(s, 1)) / DELAY(s, 1)


def alpha123(ctx):
    return (RANK(CORR(SUM((ctx.H + ctx.L) / 2, 20), SUM(MEAN(ctx.V, 60), 20), 9))
            < RANK(CORR(ctx.L, ctx.V, 6))).astype(float) * -1


def alpha124(ctx):
    return (ctx.C - ctx.VWAP) / DECAYLINEAR(RANK(TSMAX(ctx.C, 30)), 2)


def alpha125(ctx):
    return (RANK(DECAYLINEAR(CORR(ctx.VWAP, MEAN(ctx.V, 80), 17), 20))
            / RANK(DECAYLINEAR(DELTA(ctx.C * 0.5 + ctx.VWAP * 0.5, 3), 16)))


def alpha126(ctx):
    return (ctx.C + ctx.H + ctx.L) / 3


def alpha127(ctx):
    x = 100 * (ctx.C - TSMAX(ctx.C, 12)) / TSMAX(ctx.C, 12)
    return POWER(MEAN(POWER(x, 2), 12), 0.5)


def alpha128(ctx):
    tp = (ctx.H + ctx.L + ctx.C) / 3
    tpv = tp * ctx.V
    up = SUMIF(tpv, 14, tp > DELAY(tp, 1))
    dn = SUMIF(tpv, 14, tp < DELAY(tp, 1))
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def alpha129(ctx):
    d = ctx.C - DELAY(ctx.C, 1)
    return SUM(ABS(d.where(d < 0, 0.0)), 12)


def alpha130(ctx):
    return (RANK(DECAYLINEAR(CORR((ctx.H + ctx.L) / 2, MEAN(ctx.V, 40), 9), 10))
            / RANK(DECAYLINEAR(CORR(RANK(ctx.VWAP), RANK(ctx.V), 7), 3)))


def alpha131(ctx):
    return POWER(RANK(DELTA(ctx.VWAP, 1)),
                 TSRANK(CORR(ctx.C, MEAN(ctx.V, 50), 18), 18))


def alpha132(ctx):
    return MEAN(ctx.AMT, 20)


def alpha133(ctx):
    return (20 - HIGHDAY(ctx.H, 20)) / 20 * 100 - (20 - LOWDAY(ctx.L, 20)) / 20 * 100


def alpha134(ctx):
    return (ctx.C - DELAY(ctx.C, 12)) / DELAY(ctx.C, 12) * ctx.V


def alpha135(ctx):
    return SMA(DELAY(ctx.C / DELAY(ctx.C, 20), 1), 20, 1)


def alpha136(ctx):
    return (-1 * RANK(DELTA(ctx.RET, 3))) * CORR(ctx.O, ctx.V, 10)


def alpha137(ctx):
    from alphas_051_100 import _a55_core
    return _a55_core(ctx)  # Alpha55 的单日形式（无 SUM(...,20)）


def alpha138(ctx):
    return (RANK(DECAYLINEAR(DELTA(ctx.L * 0.7 + ctx.VWAP * 0.3, 3), 20))
            - TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(ctx.L, 8),
                                             TSRANK(MEAN(ctx.V, 60), 17), 5), 19), 16), 7)) * -1


def alpha139(ctx):
    return -1 * CORR(ctx.O, ctx.V, 10)


def alpha140(ctx):
    return pd.DataFrame(np.minimum(
        RANK(DECAYLINEAR(RANK(ctx.O) + RANK(ctx.L) - RANK(ctx.H) - RANK(ctx.C), 8)),
        TSRANK(DECAYLINEAR(CORR(TSRANK(ctx.C, 8), TSRANK(MEAN(ctx.V, 60), 20), 8), 7), 3)),
        index=ctx.C.index, columns=ctx.C.columns)


def alpha141(ctx):
    return RANK(CORR(RANK(ctx.H), RANK(MEAN(ctx.V, 15)), 9)) * -1


def alpha142(ctx):
    return (-1 * RANK(TSRANK(ctx.C, 10))) * RANK(DELTA(DELTA(ctx.C, 1), 1)) \
        * RANK(TSRANK(ctx.V / MEAN(ctx.V, 20), 5))


def alpha143(ctx):
    # CLOSE>DELAY(CLOSE,1) ? (CLOSE-DELAY)/DELAY*SELF : SELF；SELF=前一日因子值，初值取 1
    r = (ctx.C / DELAY(ctx.C, 1) - 1).values
    up = (ctx.C > DELAY(ctx.C, 1)).values
    out = np.ones_like(r)
    out[np.isnan(r)] = np.nan
    for t in range(1, r.shape[0]):
        prev = out[t - 1]
        cur = np.where(up[t], np.nan_to_num(r[t], nan=0.0) * np.nan_to_num(prev, nan=1.0), prev)
        cur[np.isnan(r[t])] = np.nan
        out[t] = cur
    return pd.DataFrame(out, index=ctx.C.index, columns=ctx.C.columns)


def alpha144(ctx):
    x = ABS(ctx.C / DELAY(ctx.C, 1) - 1) / ctx.AMT
    dn = ctx.C < DELAY(ctx.C, 1)
    return SUMIF(x, 20, dn) / COUNT(dn, 20).replace(0, np.nan)


def alpha145(ctx):
    return (MEAN(ctx.V, 9) - MEAN(ctx.V, 26)) / MEAN(ctx.V, 12) * 100


def alpha146(ctx):
    # 原式分母为 SMA((r-(r-SMA(r,61,2)))^2,60)，按字面化简为 SMA(r,61,2)^2 的60日均值
    # 【该因子原式存在歧义，此处为字面解读】
    r = (ctx.C - DELAY(ctx.C, 1)) / DELAY(ctx.C, 1)
    s = SMA(r, 61, 2)
    dev = r - s
    return MEAN(dev, 20) * dev / MEAN(POWER(s, 2), 60).replace(0, np.nan)


def alpha147(ctx):
    return _regbeta_seq(MEAN(ctx.C, 12), 12)


def alpha148(ctx):
    return (RANK(CORR(ctx.O, SUM(MEAN(ctx.V, 60), 9), 6))
            < RANK(ctx.O - TSMIN(ctx.O, 14))).astype(float) * -1


def alpha149(ctx):
    # REGBETA(FILTER(ret, bench_dn), FILTER(bench_ret, bench_dn), 252)
    # 仅在基准指数下跌日的样本上做滚动回归（窗口取最近252个交易日，回归样本为其中下跌日）
    ret = ctx.RET
    bret = ctx.MKT
    dn = bret < 0
    a = FILTER(ret, dn)
    b = FILTER(bret, dn)
    cov = a.rolling(252, min_periods=60).cov(b)
    var = b.rolling(252, min_periods=60).var(ddof=1)
    return cov / var.replace(0, np.nan)


def alpha150(ctx):
    return (ctx.C + ctx.H + ctx.L) / 3 * ctx.V
