# -*- coding: utf-8 -*-
"""Alpha191 因子实现（Alpha151 ~ Alpha191）"""
import numpy as np
import pandas as pd
from operators import (DELAY, DELTA, SUM, MEAN, STD, TSMAX, TSMIN, CORR, COVANCE,
                       LOG, ABS, SIGN, POWER, RANK, SMA, WMA, DECAYLINEAR,
                       TSRANK, COUNT, SUMIF, PROD, SUMAC, FMAX, FMIN,
                       HIGHDAY, LOWDAY)


def alpha151(ctx):
    return SMA(ctx.C - DELAY(ctx.C, 20), 20, 1)


def alpha152(ctx):
    x = SMA(DELAY(ctx.C / DELAY(ctx.C, 9), 1), 9, 1)
    return SMA(MEAN(DELAY(x, 1), 12) - MEAN(DELAY(x, 1), 26), 9, 1)


def alpha153(ctx):
    return (MEAN(ctx.C, 3) + MEAN(ctx.C, 6) + MEAN(ctx.C, 12) + MEAN(ctx.C, 24)) / 4


def alpha154(ctx):
    return ((ctx.VWAP - TSMIN(ctx.VWAP, 16))
            < CORR(ctx.VWAP, MEAN(ctx.V, 180), 18)).astype(float)


def alpha155(ctx):
    s13, s27 = SMA(ctx.V, 13, 2), SMA(ctx.V, 27, 2)
    return s13 - s27 - SMA(s13 - s27, 10, 2)


def alpha156(ctx):
    return pd.DataFrame(np.maximum(
        RANK(DECAYLINEAR(DELTA(ctx.VWAP, 5), 3)),
        RANK(DECAYLINEAR(-1 * DELTA(ctx.O * 0.15 + ctx.L * 0.85, 2)
                         / (ctx.O * 0.15 + ctx.L * 0.85), 3))),
        index=ctx.C.index, columns=ctx.C.columns) * -1


def alpha157(ctx):
    inner = TSMIN(RANK(RANK(-1 * RANK(DELTA(ctx.C - 1, 5)))), 2)
    p = PROD(RANK(RANK(LOG(SUM(inner, 1) + 1e-12)) + 1e-12), 5)
    left = pd.DataFrame(np.minimum(p, 5.0), index=ctx.C.index, columns=ctx.C.columns)
    return left + TSRANK(DELAY(-1 * ctx.RET, 6), 5)


def alpha158(ctx):
    s = SMA(ctx.C, 15, 2)
    return ((ctx.H - s) - (ctx.L - s)) / ctx.C


def alpha159(ctx):
    mn = TSMIN(pd.DataFrame(np.minimum(ctx.L, DELAY(ctx.C, 1)), index=ctx.C.index, columns=ctx.C.columns), 1)
    lo = pd.DataFrame(np.minimum(ctx.L, DELAY(ctx.C, 1)), index=ctx.C.index, columns=ctx.C.columns)
    hi = pd.DataFrame(np.maximum(ctx.H, DELAY(ctx.C, 1)), index=ctx.C.index, columns=ctx.C.columns)

    def leg(n):
        return (ctx.C - SUM(lo, n)) / (SUM(hi, n) - SUM(lo, n)).replace(0, np.nan)
    return (leg(6) * 12 * 24 + leg(12) * 6 * 24 + leg(24) * 6 * 24) * 100 / (6 * 12 + 6 * 24 + 12 * 24)


def alpha160(ctx):
    sd = STD(ctx.C, 20)
    return SMA(pd.DataFrame(np.where(ctx.C <= DELAY(ctx.C, 1), sd, 0.0),
                            index=ctx.C.index, columns=ctx.C.columns), 20, 1)


def _tr(ctx):
    c1 = DELAY(ctx.C, 1)
    return pd.DataFrame(np.maximum(np.maximum(ctx.H - ctx.L, ABS(c1 - ctx.H)),
                                   ABS(c1 - ctx.L)),
                        index=ctx.C.index, columns=ctx.C.columns)


def alpha161(ctx):
    return MEAN(_tr(ctx), 12)


def alpha162(ctx):
    rsi = _rsi = None
    from alphas_051_100 import _rsi_like
    rsi = _rsi_like(ctx.C, 12)
    mx, mn = TSMAX(rsi, 12), TSMIN(rsi, 12)
    return (rsi - mn) / (mx - mn).replace(0, np.nan)


def alpha163(ctx):
    return RANK((-1 * ctx.RET) * MEAN(ctx.V, 20) * ctx.VWAP * (ctx.H - ctx.C))


def alpha164(ctx):
    d = ctx.C - DELAY(ctx.C, 1)
    x = pd.DataFrame(np.where(ctx.C > DELAY(ctx.C, 1), 1 / d.where(d != 0), 1.0),
                     index=ctx.C.index, columns=ctx.C.columns)
    return SMA((x - TSMIN(x, 12)) / (ctx.H - ctx.L).replace(0, np.nan) * 100, 13, 2)


def alpha165(ctx):
    ac = SUMAC(ctx.C - MEAN(ctx.C, 48))
    return FMAX(ac) - FMIN(ac) / STD(ctx.C, 48)


def alpha166(ctx):
    r = ctx.C / DELAY(ctx.C, 1) - 1
    numer = -20 * (19 ** 1.5) * SUM(r - MEAN(r, 20), 20)
    denom = 19 * 18 * POWER(SUM(POWER(ctx.C / DELAY(ctx.C, 1), 2), 20), 1.5)
    return numer / denom.replace(0, np.nan)


def alpha167(ctx):
    d = ctx.C - DELAY(ctx.C, 1)
    return SUM(d.where(d > 0, 0.0), 12)


def alpha168(ctx):
    return -1 * ctx.V / MEAN(ctx.V, 20)


def alpha169(ctx):
    x = SMA(ctx.C - DELAY(ctx.C, 1), 9, 1)
    return SMA(MEAN(DELAY(x, 1), 12) - MEAN(DELAY(x, 1), 26), 10, 1)


def alpha170(ctx):
    return (RANK(1 / ctx.C) * ctx.V / MEAN(ctx.V, 20)
            * (ctx.H * RANK(ctx.H - ctx.C) / MEAN(ctx.H, 5))
            - RANK(ctx.VWAP - DELAY(ctx.VWAP, 5)))


def alpha171(ctx):
    return (-1 * ((ctx.L - ctx.C) * POWER(ctx.O, 5))) / ((ctx.C - ctx.H) * POWER(ctx.C, 5))


def _dmi_core(ctx):
    hd = ctx.H - DELAY(ctx.H, 1)
    ld = DELAY(ctx.L, 1) - ctx.L
    tr = _tr(ctx)
    pdm = SUM(pd.DataFrame(np.where((hd > 0) & (hd > ld), hd, 0.0),
                           index=ctx.C.index, columns=ctx.C.columns), 14)
    ndm = SUM(pd.DataFrame(np.where((ld > 0) & (ld > hd), ld, 0.0),
                           index=ctx.C.index, columns=ctx.C.columns), 14)
    trs = SUM(tr, 14).replace(0, np.nan)
    pdi, ndi = pdm * 100 / trs, ndm * 100 / trs
    return ABS(pdi - ndi) / (pdi + ndi).replace(0, np.nan) * 100


def alpha172(ctx):
    return MEAN(_dmi_core(ctx), 6)


def alpha173(ctx):
    s = SMA(ctx.C, 13, 2)
    t = SMA(SMA(SMA(LOG(ctx.C), 13, 2), 13, 2), 13, 2)
    return 3 * s - 2 * SMA(s, 13, 2) + t


def alpha174(ctx):
    sd = STD(ctx.C, 20)
    return SMA(pd.DataFrame(np.where(ctx.C > DELAY(ctx.C, 1), sd, 0.0),
                            index=ctx.C.index, columns=ctx.C.columns), 20, 1)


def alpha175(ctx):
    return MEAN(_tr(ctx), 6)


def alpha176(ctx):
    return CORR(RANK((ctx.C - TSMIN(ctx.L, 12)) / (TSMAX(ctx.H, 12) - TSMIN(ctx.L, 12))),
                RANK(ctx.V), 6)


def alpha177(ctx):
    return (20 - HIGHDAY(ctx.H, 20)) / 20 * 100


def alpha178(ctx):
    return (ctx.C - DELAY(ctx.C, 1)) / DELAY(ctx.C, 1) * ctx.V


def alpha179(ctx):
    return RANK(CORR(ctx.VWAP, ctx.V, 4)) * RANK(CORR(RANK(ctx.L), RANK(MEAN(ctx.V, 50)), 12))


def alpha180(ctx):
    cond = MEAN(ctx.V, 20) < ctx.V
    sig = SIGN(DELTA(ctx.C, 7))
    branch = -1 * TSRANK(ABS(DELTA(ctx.C, 7)), 60) * sig
    out = pd.DataFrame(np.where(cond, branch, -1 * ctx.V),
                       index=ctx.C.index, columns=ctx.C.columns)
    out[cond.isna()] = np.nan
    return out


def alpha181(ctx):
    r = ctx.C / DELAY(ctx.C, 1) - 1
    bd = ctx.BC - MEAN(ctx.BC, 20)
    return SUM((r - MEAN(r, 20)) - POWER(bd, 2), 20) / SUM(POWER(bd, 3), 20).replace(0, np.nan)


def alpha182(ctx):
    cond = ((ctx.C > ctx.O) & (ctx.BC > ctx.BO)) | ((ctx.C < ctx.O) & (ctx.BC < ctx.BO))
    return COUNT(cond, 20) / 20


def alpha183(ctx):
    ac = SUMAC(ctx.C - MEAN(ctx.C, 24))
    return FMAX(ac) - FMIN(ac) / STD(ctx.C, 24)


def alpha184(ctx):
    return RANK(CORR(DELAY(ctx.O - ctx.C, 1), ctx.C, 200)) + RANK(ctx.O - ctx.C)


def alpha185(ctx):
    return RANK(-1 * POWER(1 - ctx.O / ctx.C, 2))


def alpha186(ctx):
    x = _dmi_core(ctx)
    return (MEAN(x, 6) + DELAY(MEAN(x, 6), 6)) / 2


def alpha187(ctx):
    x = pd.DataFrame(np.where(ctx.O <= DELAY(ctx.O, 1), 0.0,
                              np.maximum(ctx.H - ctx.O, ctx.O - DELAY(ctx.O, 1))),
                     index=ctx.C.index, columns=ctx.C.columns)
    return SUM(x, 20)


def alpha188(ctx):
    r = ctx.H - ctx.L
    return (r - SMA(r, 11, 2)) / SMA(r, 11, 2).replace(0, np.nan) * 100


def alpha189(ctx):
    return MEAN(ABS(ctx.C - MEAN(ctx.C, 6)), 6)


def alpha190(ctx):
    r = ctx.C / DELAY(ctx.C, 1) - 1
    thresh = POWER(ctx.C / DELAY(ctx.C, 19), 1 / 20) - 1
    dev = r - thresh
    cnt_up = COUNT(dev > 0, 20)
    cnt_dn = COUNT(dev < 0, 20)
    sum_dn = SUMIF(POWER(dev, 2), 20, dev < 0)
    sum_up = SUMIF(POWER(dev, 2), 20, dev > 0)
    inner = ((cnt_up - 1) * sum_dn) / (cnt_dn * sum_up).replace(0, np.nan)
    return LOG(inner)


def alpha191(ctx):
    return (CORR(MEAN(ctx.V, 20), ctx.L, 5) + (ctx.H + ctx.L) / 2) - ctx.C
