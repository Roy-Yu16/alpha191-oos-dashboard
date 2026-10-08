# -*- coding: utf-8 -*-
"""Alpha191 因子实现（Alpha51 ~ Alpha100）。
Alpha51-58 为报告 OCR 缺失条目，按 BigQuant 公开转录版补齐。"""
import numpy as np
import pandas as pd
from operators import (DELAY, DELTA, SUM, MEAN, STD, TSMAX, TSMIN, CORR, COVANCE,
                       LOG, ABS, SIGN, POWER, RANK, SMA, WMA, DECAYLINEAR,
                       TSRANK, COUNT, SUMIF, PROD)
from alphas_001_050 import _a3_core, _a11_core, _a49_terms


def alpha051(ctx):
    # 公开转录版: SUM(严格上移gap,12)/(SUM(上移,12)+SUM(下移,12)) —— 与 Alpha49 方向相反
    dn, up = _a49_terms(ctx)
    return up / (dn + up)


def alpha052(ctx):
    tp = (ctx.H + ctx.L + ctx.C) / 3
    tp1 = DELAY(tp, 1)
    up = SUM(pd.DataFrame(np.maximum(0.0, ctx.H - tp1), index=ctx.C.index, columns=ctx.C.columns), 26)
    dn = SUM(pd.DataFrame(np.maximum(0.0, tp1 - ctx.L), index=ctx.C.index, columns=ctx.C.columns), 26)
    return up / dn.replace(0, np.nan) * 100


def alpha053(ctx):
    return COUNT(ctx.C > DELAY(ctx.C, 1), 12) / 12 * 100


def alpha054(ctx):
    return -1 * RANK(STD(ABS(ctx.C - ctx.O), 5) + (ctx.C - ctx.O) + CORR(ctx.C, ctx.O, 10))


def _a55_core(ctx):
    c1, o1 = DELAY(ctx.C, 1), DELAY(ctx.O, 1)
    ah, al = ABS(ctx.H - c1), ABS(ctx.L - c1)
    ahl = ABS(ctx.H - DELAY(ctx.L, 1))
    adc = ABS(c1 - o1)
    numer = 16 * (ctx.C - c1 + (ctx.C - ctx.O) / 2 + c1 - o1)
    denom = pd.DataFrame(np.select(
        [(ah > al) & (ah > ahl), (al > ahl) & (al > ah)],
        [ah + al / 2 + adc / 4, al + ah / 2 + adc / 4],
        default=ahl + adc / 4), index=ctx.C.index, columns=ctx.C.columns)
    return numer / denom.replace(0, np.nan) * pd.DataFrame(
        np.maximum(ah, al), index=ctx.C.index, columns=ctx.C.columns)


def alpha055(ctx):
    return SUM(_a55_core(ctx), 20)


def alpha056(ctx):
    left = RANK(ctx.O - TSMIN(ctx.O, 12))
    right = RANK(POWER(RANK(CORR(SUM((ctx.H + ctx.L) / 2, 19),
                                 SUM(MEAN(ctx.V, 40), 19), 13)), 5))
    return (left < right).astype(float).where(left.notna() & right.notna())


def alpha057(ctx):
    return SMA((ctx.C - TSMIN(ctx.L, 9)) / (TSMAX(ctx.H, 9) - TSMIN(ctx.L, 9)) * 100, 3, 1)


def alpha058(ctx):
    return COUNT(ctx.C > DELAY(ctx.C, 1), 20) / 20 * 100


def alpha059(ctx):
    return _a3_core(ctx, 20)


def alpha060(ctx):
    return _a11_core(ctx, 20)


def alpha061(ctx):
    return pd.DataFrame(np.maximum(
        RANK(DECAYLINEAR(DELTA(ctx.VWAP, 1), 12)),
        RANK(DECAYLINEAR(RANK(CORR(ctx.L, MEAN(ctx.V, 80), 8)), 17))),
        index=ctx.C.index, columns=ctx.C.columns) * -1


def alpha062(ctx):
    return -1 * CORR(ctx.H, RANK(ctx.V), 5)


def _rsi_like(x, n):
    d = x - DELAY(x, 1)
    up = SMA(pd.DataFrame(np.maximum(d, 0.0), index=x.index, columns=x.columns), n, 1)
    ad = SMA(ABS(d), n, 1)
    return up / ad.replace(0, np.nan) * 100


def alpha063(ctx):
    return _rsi_like(ctx.C, 6)


def alpha064(ctx):
    return pd.DataFrame(np.maximum(
        RANK(DECAYLINEAR(CORR(RANK(ctx.VWAP), RANK(ctx.V), 4), 4)),
        RANK(DECAYLINEAR(TSMAX(CORR(RANK(ctx.C), RANK(MEAN(ctx.V, 60)), 4), 13), 14))),
        index=ctx.C.index, columns=ctx.C.columns) * -1


def alpha065(ctx):
    return MEAN(ctx.C, 6) / ctx.C


def alpha066(ctx):
    return (ctx.C - MEAN(ctx.C, 6)) / MEAN(ctx.C, 6) * 100


def alpha067(ctx):
    return _rsi_like(ctx.C, 24)


def alpha068(ctx):
    return SMA(((ctx.H + ctx.L) / 2 - (DELAY(ctx.H, 1) + DELAY(ctx.L, 1)) / 2)
               * (ctx.H - ctx.L) / ctx.V, 15, 2)


def alpha069(ctx):
    dtm = pd.DataFrame(np.where(ctx.O <= DELAY(ctx.O, 1), 0.0,
                                np.maximum(ctx.H - ctx.O, ctx.O - DELAY(ctx.O, 1))),
                       index=ctx.C.index, columns=ctx.C.columns)
    dbm = pd.DataFrame(np.where(ctx.O >= DELAY(ctx.O, 1), 0.0,
                                np.maximum(ctx.O - ctx.L, ctx.O - DELAY(ctx.O, 1))),
                       index=ctx.C.index, columns=ctx.C.columns)
    sd, sb = SUM(dtm, 20), SUM(dbm, 20)
    out = pd.DataFrame(np.select(
        [sd > sb, sd == sb],
        [(sd - sb) / sd, 0.0], default=(sd - sb) / sb.replace(0, np.nan)),
        index=ctx.C.index, columns=ctx.C.columns)
    out[sd.isna()] = np.nan
    return out


def alpha070(ctx):
    return STD(ctx.AMT, 6)


def alpha071(ctx):
    return (ctx.C - MEAN(ctx.C, 24)) / MEAN(ctx.C, 24) * 100


def alpha072(ctx):
    return SMA((TSMAX(ctx.H, 6) - ctx.C) / (TSMAX(ctx.H, 6) - TSMIN(ctx.L, 6)) * 100, 15, 1)


def alpha073(ctx):
    return (TSRANK(DECAYLINEAR(DECAYLINEAR(CORR(ctx.C, ctx.V, 10), 16), 4), 5)
            - RANK(DECAYLINEAR(CORR(ctx.VWAP, MEAN(ctx.V, 30), 4), 3))) * -1


def alpha074(ctx):
    return (RANK(CORR(SUM(ctx.L * 0.35 + ctx.VWAP * 0.65, 20),
                      SUM(MEAN(ctx.V, 40), 20), 7))
            + RANK(CORR(RANK(ctx.VWAP), RANK(ctx.V), 6)))


def alpha075(ctx):
    cond_dn = ctx.BC < ctx.BO
    return COUNT((ctx.C > ctx.O) & cond_dn, 50) / COUNT(cond_dn, 50).replace(0, np.nan)


def alpha076(ctx):
    x = ABS(ctx.C / DELAY(ctx.C, 1) - 1) / ctx.V
    return STD(x, 20) / MEAN(x, 20)


def alpha077(ctx):
    return pd.DataFrame(np.minimum(
        RANK(DECAYLINEAR((ctx.H + ctx.L) / 2 + ctx.H - (ctx.VWAP + ctx.H), 20)),
        RANK(DECAYLINEAR(CORR((ctx.H + ctx.L) / 2, MEAN(ctx.V, 40), 3), 6))),
        index=ctx.C.index, columns=ctx.C.columns)


def alpha078(ctx):
    tp = (ctx.H + ctx.L + ctx.C) / 3
    ma = MEAN(tp, 12)
    md = MEAN(ABS(ctx.C - ma), 12)  # 按原文: MEAN(ABS(CLOSE-MEAN(tp,12)),12)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def alpha079(ctx):
    return _rsi_like(ctx.C, 12)


def alpha080(ctx):
    return (ctx.V - DELAY(ctx.V, 5)) / DELAY(ctx.V, 5) * 100


def alpha081(ctx):
    return SMA(ctx.V, 21, 2)


def alpha082(ctx):
    return SMA((TSMAX(ctx.H, 6) - ctx.C) / (TSMAX(ctx.H, 6) - TSMIN(ctx.L, 6)) * 100, 20, 1)


def alpha083(ctx):
    return -1 * RANK(COVANCE(RANK(ctx.H), RANK(ctx.V), 5))


def alpha084(ctx):
    return SUM(pd.DataFrame(np.where(ctx.C > DELAY(ctx.C, 1), ctx.V,
                                     np.where(ctx.C < DELAY(ctx.C, 1), -ctx.V, 0.0)),
                            index=ctx.C.index, columns=ctx.C.columns), 20)


def alpha085(ctx):
    return TSRANK(ctx.V / MEAN(ctx.V, 20), 20) * TSRANK(-1 * DELTA(ctx.C, 7), 8)


def alpha086(ctx):
    x = (DELAY(ctx.C, 20) - DELAY(ctx.C, 10)) / 10 - (DELAY(ctx.C, 10) - ctx.C) / 10
    out = pd.DataFrame(np.select([x > 0.25, x < 0],
                                 [-1.0, 1.0],
                                 default=(-1 * (ctx.C - DELAY(ctx.C, 1))).values),
                       index=ctx.C.index, columns=ctx.C.columns)
    out[x.isna()] = np.nan
    return out


def alpha087(ctx):
    return (RANK(DECAYLINEAR(DELTA(ctx.VWAP, 4), 7))
            + TSRANK(DECAYLINEAR((ctx.L - ctx.VWAP) / (ctx.O - (ctx.H + ctx.L) / 2), 11), 7)) * -1


def alpha088(ctx):
    return (ctx.C - DELAY(ctx.C, 20)) / DELAY(ctx.C, 20) * 100


def alpha089(ctx):
    s13, s27 = SMA(ctx.C, 13, 2), SMA(ctx.C, 27, 2)
    return 2 * (s13 - s27 - SMA(s13 - s27, 10, 2))


def alpha090(ctx):
    return RANK(CORR(RANK(ctx.VWAP), RANK(ctx.V), 5)) * -1


def alpha091(ctx):
    return RANK(ctx.C - TSMAX(ctx.C, 5)) * RANK(CORR(MEAN(ctx.V, 40), ctx.L, 5)) * -1


def alpha092(ctx):
    return pd.DataFrame(np.maximum(
        RANK(DECAYLINEAR(DELTA(ctx.C * 0.35 + ctx.VWAP * 0.65, 2), 3)),
        TSRANK(DECAYLINEAR(ABS(CORR(MEAN(ctx.V, 180), ctx.C, 13)), 5), 15)),
        index=ctx.C.index, columns=ctx.C.columns) * -1


def alpha093(ctx):
    x = pd.DataFrame(np.where(ctx.O >= DELAY(ctx.O, 1), 0.0,
                              np.maximum(ctx.O - ctx.L, ctx.O - DELAY(ctx.O, 1))),
                     index=ctx.C.index, columns=ctx.C.columns)
    return SUM(x, 20)


def alpha094(ctx):
    return SUM(pd.DataFrame(np.where(ctx.C > DELAY(ctx.C, 1), ctx.V,
                                     np.where(ctx.C < DELAY(ctx.C, 1), -ctx.V, 0.0)),
                            index=ctx.C.index, columns=ctx.C.columns), 30)


def alpha095(ctx):
    return STD(ctx.AMT, 20)


def alpha096(ctx):
    k = SMA((ctx.C - TSMIN(ctx.L, 9)) / (TSMAX(ctx.H, 9) - TSMIN(ctx.L, 9)) * 100, 3, 1)
    return SMA(k, 3, 1)


def alpha097(ctx):
    return STD(ctx.V, 10)


def alpha098(ctx):
    x = DELTA(MEAN(ctx.C, 100), 100) / DELAY(ctx.C, 100)
    out = pd.DataFrame(np.where(x <= 0.05,
                                -1 * (ctx.C - TSMIN(ctx.C, 100)),
                                -1 * DELTA(ctx.C, 3)),
                       index=ctx.C.index, columns=ctx.C.columns)
    out[x.isna()] = np.nan
    return out


def alpha099(ctx):
    return -1 * RANK(COVANCE(RANK(ctx.C), RANK(ctx.V), 5))


def alpha100(ctx):
    return STD(ctx.V, 20)
