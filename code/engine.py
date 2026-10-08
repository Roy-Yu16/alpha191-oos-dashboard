# -*- coding: utf-8 -*-
"""
Alpha191 引擎：数据装配 + 全部因子计算
数据约定（与用户本地库 export_csi500.py 输出对齐）:
  daily parquet 列: date, code, close(不复权), qfq_open/high/low/close(前复权),
                    volume(股), amount(元), pct_chg, limit_up, limit_down
  index parquet 列: date, open, close（基准指数，默认中证500 000905）
价格口径：OPEN/HIGH/LOW/CLOSE 用前复权价；VWAP=AMOUNT/VOLUME 再乘复权因子；
          VOLUME/AMOUNT 用原始值；RET=前复权收盘涨跌幅。
"""
import time
import traceback
from types import SimpleNamespace

import numpy as np
import pandas as pd

import alphas_001_050 as m1
import alphas_051_100 as m2
import alphas_101_150 as m3
import alphas_151_191 as m4

REGISTRY = {}
for m in (m1, m2, m3, m4):
    for name in dir(m):
        if name.startswith('alpha') and name != 'alphas':
            try:
                idx = int(name[5:])
                REGISTRY[idx] = getattr(m, name)
            except ValueError:
                pass
assert len(REGISTRY) == 191, f"registry size {len(REGISTRY)}"


def build_ctx(daily: pd.DataFrame, bench: pd.DataFrame) -> SimpleNamespace:
    """把长表日线装配成宽表矩阵上下文"""
    d = daily.copy()
    d['date'] = pd.to_datetime(d['date'])
    # 复权因子与 VWAP
    d['adj'] = d['qfq_close'] / d['close']
    d['vwap'] = d['amount'] / d['volume'].replace(0, np.nan) * d['adj']

    def wide(col):
        w = d.pivot(index='date', columns='code', values=col)
        return w.astype('float64').sort_index()

    O, H, L = wide('qfq_open'), wide('qfq_high'), wide('qfq_low')
    C = wide('qfq_close')
    V = wide('volume')
    AMT = wide('amount')
    VWAP = wide('vwap')
    RET = C.pct_change(fill_method=None)  # 停牌期不产生虚假收益

    # 基准指数（广播到所有列）
    b = bench.copy()
    b['date'] = pd.to_datetime(b['date'])
    b = b.set_index('date').sort_index()
    b = b.reindex(C.index)
    bclose = b['close'].astype(float)
    bopen = b['open'].astype(float)
    BC = pd.DataFrame(np.tile(bclose.values[:, None], (1, C.shape[1])),
                      index=C.index, columns=C.columns)
    BO = pd.DataFrame(np.tile(bopen.values[:, None], (1, C.shape[1])),
                      index=C.index, columns=C.columns)
    MKT = BC / BC.shift(1) - 1

    return SimpleNamespace(O=O, H=H, L=L, C=C, VWAP=VWAP, V=V, AMT=AMT,
                           RET=RET, BO=BO, BC=BC, MKT=MKT)


def run_all(ctx, out_dir=None, ids=None, verbose=True):
    """计算全部因子，返回 {id: wide DataFrame}；out_dir 给定则每个因子存 parquet"""
    results, errors = {}, {}
    for i in sorted(REGISTRY):
        if ids and i not in ids:
            continue
        t0 = time.time()
        try:
            f = REGISTRY[i](ctx)
            f = f.replace([np.inf, -np.inf], np.nan)
            results[i] = f
            if out_dir:
                f.astype('float32').to_parquet(f"{out_dir}/alpha{i:03d}.parquet")
            if verbose:
                cov = f.notna().sum().sum()
                print(f"alpha{i:03d} ok  ({time.time()-t0:.1f}s, 非空 {cov})")
        except Exception as e:
            errors[i] = traceback.format_exc()
            print(f"alpha{i:03d} FAIL: {e}")
    return results, errors
