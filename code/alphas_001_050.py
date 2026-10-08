# -*- coding: utf-8 -*-
"""
Alpha191 因子实现（Alpha1 ~ Alpha50）
公式来源：国泰君安《基于短周期价量特征的多因子选股体系》(2017-06-15) 表6，
          缺失/乱码条目以 BigQuant 收录的公开转录版校对补齐（Alpha51-58、Alpha149）。
ctx 字段: O,H,L,C,VWAP,V,AMT,RET 为宽表(date x code); BO,BC 为基准指数开/收宽表;
          MKT 为基准日收益宽表。
"""
import numpy as np
import pandas as pd
from operators import (DELAY, DELTA, SUM, MEAN, STD, TSMAX, TSMIN, CORR, COVANCE,
                       CORR_FULL, LOG, ABS, SIGN, POWER, RANK, SMA, SMEAN, WMA,
                       DECAYLINEAR, TSRANK, COUNT, SUMIF, FILTER, PROD, SUMAC,
                       FMAX, FMIN, HIGHDAY, LOWDAY, REGBETA, REGRESI)


def _regbeta_seq(y, n):
    """REGBETA(y, SEQUENCE(n))：y 对 1..n 序列的滚动回归斜率"""
    t = np.arange(1, n + 1, dtype=float)
    tm, tv = t.mean(), t.var(ddof=1)
    # slope = (E[ty]-tm*E[y]) / var(t)，滚动 E[ty] 用卷积
    ty = y.rolling(n, min_periods=n).apply(lambda v: np.dot(v, t) / n, raw=True)
    my = y.rolling(n, min_periods=n).mean()
    return (ty - tm * my) / tv


def alpha001(ctx):
    return -1 * CORR(RANK(DELTA(LOG(ctx.V), 1)), RANK((ctx.C - ctx.O) / ctx.O), 6)


def alpha002(ctx):
    return -1 * DELTA(((ctx.C - ctx.L) - (ctx.H - ctx.C)) / (ctx.H - ctx.L), 1)


def _a3_core(ctx, n):
    c, cd = ctx.C, DELAY(ctx.C, 1)
    part = pd.DataFrame(np.where(c == cd, 0.0,
                                 c - np.where(c > cd, np.minimum(ctx.L, cd),
                                              np.maximum(ctx.H, cd))),
                        index=c.index, columns=c.columns)
    return SUM(part, n)


def alpha003(ctx):
    return _a3_core(ctx, 6)


def alpha004(ctx):
    m8, s8, m2 = MEAN(ctx.C, 8), STD(ctx.C, 8), MEAN(ctx.C, 2)
    cond_hi = (m8 + s8) < m2
    cond_lo = m2 < (m8 - s8)
    cond_vol = (ctx.V / MEAN(ctx.V, 20)) >= 1
    out = pd.DataFrame(np.select([cond_hi, cond_lo], [-1.0, 1.0],
                                 default=np.where(cond_vol, 1.0, -1.0)),
                       index=ctx.C.index, columns=ctx.C.columns)
    out[(m8.isna()) | (MEAN(ctx.V, 20).isna())] = np.nan
    return out


def alpha005(ctx):
    return -1 * TSMAX(CORR(TSRANK(ctx.V, 5), TSRANK(ctx.H, 5), 5), 3)


def alpha006(ctx):
    return RANK(SIGN(DELTA(ctx.O * 0.85 + ctx.H * 0.15, 4))) * -1


def alpha007(ctx):
    return (RANK(TSMAX(ctx.VWAP - ctx.C, 3)) + RANK(TSMIN(ctx.VWAP - ctx.C, 3))) \
        * RANK(DELTA(ctx.V, 3))


def alpha008(ctx):
    return RANK(DELTA((ctx.H + ctx.L) / 2 * 0.2 + ctx.VWAP * 0.8, 4)) * -1


def alpha009(ctx):
    return SMA(((ctx.H + ctx.L) / 2 - (DELAY(ctx.H, 1) + DELAY(ctx.L, 1)) / 2)
               * (ctx.H - ctx.L) / ctx.V, 7, 2)


def alpha010(ctx):
    return RANK(TSMAX(POWER(pd.DataFrame(np.where(ctx.RET < 0, STD(ctx.RET, 20), ctx.C),
                                         index=ctx.C.index, columns=ctx.C.columns), 2), 5))


def _a11_core(ctx, n):
    return SUM(((ctx.C - ctx.L) - (ctx.H - ctx.C)) / (ctx.H - ctx.L) * ctx.V, n)


def alpha011(ctx):
    return _a11_core(ctx, 6)


def alpha012(ctx):
    return RANK(ctx.O - MEAN(ctx.VWAP, 10)) * (-1 * RANK(ABS(ctx.C - ctx.VWAP)))


def alpha013(ctx):
    return POWER(ctx.H * ctx.L, 0.5) - ctx.VWAP


def alpha014(ctx):
    return ctx.C - DELAY(ctx.C, 5)


def alpha015(ctx):
    return ctx.O / DELAY(ctx.C, 1) - 1


def alpha016(ctx):
    return -1 * TSMAX(RANK(CORR(RANK(ctx.V), RANK(ctx.VWAP), 5)), 5)


def alpha017(ctx):
    return POWER(RANK(ctx.VWAP - TSMAX(ctx.VWAP, 15)), DELTA(ctx.C, 5))


def alpha018(ctx):
    return ctx.C / DELAY(ctx.C, 5)


def alpha019(ctx):
    cd = DELAY(ctx.C, 5)
    out = pd.DataFrame(np.select(
        [ctx.C < cd, ctx.C == cd],
        [(ctx.C - cd) / cd, 0.0], default=(ctx.C - cd) / ctx.C),
        index=ctx.C.index, columns=ctx.C.columns)
    out[cd.isna()] = np.nan
    return out


def alpha020(ctx):
    return (ctx.C - DELAY(ctx.C, 6)) / DELAY(ctx.C, 6) * 100


def alpha021(ctx):
    return _regbeta_seq(MEAN(ctx.C, 6), 6)


def alpha022(ctx):
    dev = (ctx.C - MEAN(ctx.C, 6)) / MEAN(ctx.C, 6)
    return SMEAN(dev - DELAY(dev, 3), 12, 1)


def alpha023(ctx):
    sd = STD(ctx.C, 20)
    up = SMA(pd.DataFrame(np.where(ctx.C > DELAY(ctx.C, 1), sd, 0.0),
                          index=ctx.C.index, columns=ctx.C.columns), 20, 1)
    dn = SMA(pd.DataFrame(np.where(ctx.C <= DELAY(ctx.C, 1), sd, 0.0),
                          index=ctx.C.index, columns=ctx.C.columns), 20, 1)
    return up / (up + dn) * 100


def alpha024(ctx):
    return SMA(ctx.C - DELAY(ctx.C, 5), 5, 1)


def alpha025(ctx):
    return (-1 * RANK(DELTA(ctx.C, 7)
                      * (1 - RANK(DECAYLINEAR(ctx.V / MEAN(ctx.V, 20), 9))))) \
        * (1 + RANK(SUM(ctx.RET, 250)))


def alpha026(ctx):
    return (MEAN(ctx.C, 7) - ctx.C) + CORR(ctx.VWAP, DELAY(ctx.C, 5), 230)


def alpha027(ctx):
    return WMA((ctx.C - DELAY(ctx.C, 3)) / DELAY(ctx.C, 3) * 100
               + (ctx.C - DELAY(ctx.C, 6)) / DELAY(ctx.C, 6) * 100, 12)


def alpha028(ctx):
    k = SMA((ctx.C - TSMIN(ctx.L, 9)) / (TSMAX(ctx.H, 9) - TSMIN(ctx.L, 9)) * 100, 3, 1)
    return 3 * k - 2 * SMA(k, 3, 1)


def alpha029(ctx):
    return (ctx.C - DELAY(ctx.C, 6)) / DELAY(ctx.C, 6) * ctx.V


def alpha030(ctx):
    # 原公式: WMA((REGRESI(CLOSE/DELAY(CLOSE)-1, MKT, SMB, HML, 60))^2, 20)
    # SMB/HML 因子不可得（附录2注明为 Fama-French 三因子）→ 替代实现：仅对市场收益回归取残差
    # 【替代口径，非原样复现】
    resid = REGRESI(ctx.C / DELAY(ctx.C, 1) - 1, ctx.MKT, 60)
    return WMA(POWER(resid, 2), 20)


def alpha031(ctx):
    return (ctx.C - MEAN(ctx.C, 12)) / MEAN(ctx.C, 12) * 100


def alpha032(ctx):
    return -1 * SUM(RANK(CORR(RANK(ctx.H), RANK(ctx.V), 3)), 3)


def alpha033(ctx):
    return ((-1 * TSMIN(ctx.L, 5) + DELAY(TSMIN(ctx.L, 5), 5))
            * RANK((SUM(ctx.RET, 240) - SUM(ctx.RET, 20)) / 220)) * TSRANK(ctx.V, 5)


def alpha034(ctx):
    return MEAN(ctx.C, 12) / ctx.C


def alpha035(ctx):
    return (pd.DataFrame(np.minimum(RANK(DECAYLINEAR(DELTA(ctx.O, 1), 15)),
                                    RANK(DECAYLINEAR(CORR(ctx.V, ctx.O, 17), 7))),
                         index=ctx.C.index, columns=ctx.C.columns)) * -1


def alpha036(ctx):
    return RANK(SUM(CORR_FULL(RANK(ctx.V), RANK(ctx.VWAP)), 6))


def alpha037(ctx):
    return -1 * RANK(SUM(ctx.O, 5) * SUM(ctx.RET, 5)
                     - DELAY(SUM(ctx.O, 5) * SUM(ctx.RET, 5), 10))


def alpha038(ctx):
    cond = MEAN(ctx.H, 20) < ctx.H
    return pd.DataFrame(np.where(cond, -1 * DELTA(ctx.H, 2), 0.0),
                        index=ctx.C.index, columns=ctx.C.columns)


def alpha039(ctx):
    return (RANK(DECAYLINEAR(DELTA(ctx.C, 2), 8))
            - RANK(DECAYLINEAR(CORR(ctx.VWAP * 0.3 + ctx.O * 0.7,
                                    SUM(MEAN(ctx.V, 180), 37), 14), 12))) * -1


def alpha040(ctx):
    up = SUMIF(ctx.V, 26, ctx.C > DELAY(ctx.C, 1))
    dn = SUMIF(ctx.V, 26, ctx.C <= DELAY(ctx.C, 1))
    return up / dn.replace(0, np.nan) * 100


def alpha041(ctx):
    return RANK(TSMAX(DELTA(ctx.VWAP, 3), 5)) * -1


def alpha042(ctx):
    return (-1 * RANK(STD(ctx.H, 10))) * CORR(ctx.H, ctx.V, 10)


def alpha043(ctx):
    return SUM(pd.DataFrame(np.where(ctx.C > DELAY(ctx.C, 1), ctx.V,
                                     np.where(ctx.C < DELAY(ctx.C, 1), -ctx.V, 0.0)),
                            index=ctx.C.index, columns=ctx.C.columns), 6)


def alpha044(ctx):
    return (TSRANK(DECAYLINEAR(CORR(ctx.L, MEAN(ctx.V, 10), 7), 6), 4)
            + TSRANK(DECAYLINEAR(DELTA(ctx.VWAP, 3), 10), 15))


def alpha045(ctx):
    return RANK(DELTA(ctx.C * 0.6 + ctx.O * 0.4, 1)) \
        * RANK(CORR(ctx.VWAP, MEAN(ctx.V, 150), 15))


def alpha046(ctx):
    return (MEAN(ctx.C, 3) + MEAN(ctx.C, 6) + MEAN(ctx.C, 12) + MEAN(ctx.C, 24)) \
        / (4 * ctx.C)


def alpha047(ctx):
    return SMA((TSMAX(ctx.H, 6) - ctx.C) / (TSMAX(ctx.H, 6) - TSMIN(ctx.L, 6)) * 100, 9, 1)


def alpha048(ctx):
    s = (SIGN(ctx.C - DELAY(ctx.C, 1)) + SIGN(DELAY(ctx.C, 1) - DELAY(ctx.C, 2))
         + SIGN(DELAY(ctx.C, 2) - DELAY(ctx.C, 3)))
    return -1 * (RANK(s) * SUM(ctx.V, 5)) / SUM(ctx.V, 20)


def _a49_terms(ctx):
    h1, l1 = DELAY(ctx.H, 1), DELAY(ctx.L, 1)
    gap = pd.DataFrame(np.maximum(ABS(ctx.H - h1), ABS(ctx.L - l1)),
                       index=ctx.C.index, columns=ctx.C.columns)
    down = gap.where((ctx.H + ctx.L) < (h1 + l1), 0.0)   # 整体下移
    up = gap.where((ctx.H + ctx.L) > (h1 + l1), 0.0)     # 整体上移
    return SUM(down, 12), SUM(up, 12)


def alpha049(ctx):
    # 注：报告原式为严格不等号；等号情形归入"下移"（与 BigQuant 转录版一致）
    dn, up = _a49_terms(ctx)
    return dn / (dn + up)


def alpha050(ctx):
    dn, up = _a49_terms(ctx)
    return up / (dn + up) - dn / (dn + up)
