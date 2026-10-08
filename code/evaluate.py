# -*- coding: utf-8 -*-
"""
因子评价模块：
1) PIT 中证500成分掩码（membership 区间表，月度粒度）
2) 前瞻收益（1/2/3/5/10 日）
3) RankIC / ICIR / t 值（原始 + 行业&规模中性化）
4) 十分位分层多空（每5日非重叠调仓）
5) 分年度统计
"""
import numpy as np
import pandas as pd


# ---------------- 成分掩码 ----------------
def membership_mask(mem: pd.DataFrame, dates: pd.DatetimeIndex, codes) -> pd.DataFrame:
    """返回 date x code 的 bool 掩码：当日是否为中证500成分（in_date <= t < out_date）"""
    m = mem.copy()
    m['in_date'] = pd.to_datetime(m['in_date'])
    m['out_date'] = pd.to_datetime(m['out_date'])
    codes = list(codes)
    mask = pd.DataFrame(False, index=dates, columns=codes)
    dvals = dates.values
    for _, row in m.iterrows():
        c = str(row['code']).zfill(6)
        if c not in mask.columns:
            continue
        lo = np.searchsorted(dvals, np.datetime64(row['in_date']))
        if pd.isna(row['out_date']):
            hi = len(dvals)
        else:
            hi = np.searchsorted(dvals, np.datetime64(row['out_date']))
        if lo < hi:
            mask.iloc[lo:hi, mask.columns.get_loc(c)] = True
    return mask


# ---------------- 前瞻收益 ----------------
def forward_returns(close: pd.DataFrame, horizons=(1, 2, 3, 5, 10)):
    return {d: close.shift(-d) / close - 1 for d in horizons}


# ---------------- 中性化 ----------------
def neutralize(factor: pd.DataFrame, industry: pd.DataFrame, lncap: pd.DataFrame) -> pd.DataFrame:
    """逐日截面回归：factor ~ 行业哑变量 + ln流通市值，取残差。
    industry/lncap 均为与 factor 同形的宽表（industry 为行业代码/名称）。"""
    out = pd.DataFrame(np.nan, index=factor.index, columns=factor.columns)
    for t in factor.index:
        f = factor.loc[t]
        ind = industry.loc[t]
        cap = lncap.loc[t]
        ok = f.notna() & ind.notna() & cap.notna()
        if ok.sum() < 40:
            continue
        fv = f[ok].values.astype(float)
        dummies = pd.get_dummies(ind[ok]).values.astype(float)
        X = np.column_stack([dummies, cap[ok].values])
        # 岭化最小二乘防共线
        beta, *_ = np.linalg.lstsq(X.T @ X + 1e-8 * np.eye(X.shape[1]), X.T @ fv,
                                   rcond=None)
        out.loc[t, ok[ok].index] = fv - X @ beta
    return out


# ---------------- RankIC ----------------
def rank_ic(factor: pd.DataFrame, fwd: pd.DataFrame, mask: pd.DataFrame) -> pd.Series:
    """逐日 Spearman 秩相关（mask 内）"""
    f = factor.where(mask)
    r = fwd.where(mask)
    fr = f.rank(axis=1)
    rr = r.rank(axis=1)
    fdm = fr.sub(fr.mean(axis=1), axis=0)
    rdm = rr.sub(rr.mean(axis=1), axis=0)
    num = (fdm * rdm).sum(axis=1, min_count=30)
    den = np.sqrt((fdm ** 2).sum(axis=1) * (rdm ** 2).sum(axis=1))
    ic = num / den.replace(0, np.nan)
    cnt = f.notna().sum(axis=1)
    ic[cnt < 30] = np.nan
    return ic


def ic_stats(ic: pd.Series, periods_per_year=244) -> dict:
    ic = ic.dropna()
    if len(ic) < 20:
        return {}
    mean, std = ic.mean(), ic.std()
    return {
        'IC均值': mean, 'IC标准差': std,
        'ICIR': mean / std if std > 0 else np.nan,
        't值': mean / std * np.sqrt(len(ic)) if std > 0 else np.nan,
        'IC>0占比': (ic > 0).mean(),
        '样本期数': len(ic),
    }


# ---------------- 十分位分层 ----------------
def decile_longshort(factor: pd.DataFrame, close: pd.DataFrame, mask: pd.DataFrame,
                     hold=5, n_group=10, cost=0.0):
    """每 hold 日调仓的非重叠十分位组合。
    返回: dict(group_returns=DataFrame[date x group], ls=Series 多空日收益, nav=...)、换手率"""
    dates = factor.index
    start_idx = np.argmax(mask.sum(axis=1).values >= 100)  # 成分数充足起点
    reb_dates = dates[start_idx::hold]
    grp_ret = {g: [] for g in range(1, n_group + 1)}
    grp_dates, turnovers = [], []
    prev_w = None
    for t in reb_dates:
        pos = dates.get_loc(t)
        if pos + hold >= len(dates):
            break
        f = factor.loc[t].where(mask.loc[t])
        f = f.dropna()
        if len(f) < 100:
            continue
        q = pd.qcut(f.rank(method='first'), n_group, labels=False) + 1  # 1=最小 .. 10=最大
        # 等权持有 hold 日（用日收益复利）
        seg = close.iloc[pos + 1: pos + hold + 1] / close.iloc[pos:pos + hold].values - 1
        for g in range(1, n_group + 1):
            cols = q[q == g].index
            if len(cols) == 0:
                grp_ret[g].append(np.nan)
                continue
            daily = seg[cols].mean(axis=1)          # 组内等权日收益
            grp_ret[g].append((1 + daily).prod() - 1)
        w = pd.Series(1.0, index=q[q == n_group].index)  # 多头组权重（用于换手）
        if prev_w is not None:
            allidx = prev_w.index.union(w.index)
            turnovers.append((w.reindex(allidx, fill_value=0)
                              - prev_w.reindex(allidx, fill_value=0)).abs().sum())
        prev_w = w
        grp_dates.append(t)
    gret = pd.DataFrame({g: pd.Series(v, index=grp_dates[:len(v)])
                         for g, v in grp_ret.items()})
    ls = gret[n_group] - gret[1]
    if cost > 0 and len(turnovers):
        ls = ls - cost * (pd.Series(turnovers, index=ls.index[1:]).reindex(ls.index).fillna(1.0) + 1.0)
    nav = (1 + ls.fillna(0)).cumprod()
    return {'group_returns': gret, 'ls': ls, 'nav': nav,
            'turnover_single_side_mean': np.mean(turnovers) / 2 if turnovers else np.nan,
            'rebalance_dates': pd.DatetimeIndex(grp_dates)}


def ls_stats(ls: pd.Series, hold=5, periods_per_year=244) -> dict:
    ls = ls.dropna()
    if len(ls) < 10:
        return {}
    ppy = periods_per_year / hold
    nav = (1 + ls).cumprod()
    ann = nav.iloc[-1] ** (ppy / len(ls)) - 1
    vol = ls.std() * np.sqrt(ppy)
    dd = (nav / nav.cummax() - 1).min()
    return {'年化多空收益': ann, '年化波动': vol, '夏普/IR': ann / vol if vol > 0 else np.nan,
            '最大回撤': dd, '胜率': (ls > 0).mean(), '期数': len(ls)}
