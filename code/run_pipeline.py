# -*- coding: utf-8 -*-
"""191因子全量计算 + 评价流水线（在Kimi沙箱中运行）"""
import os, sys, time, traceback
import numpy as np
import pandas as pd

BASE = '/mnt/agents/output/work'
sys.path.insert(0, f'{BASE}/alpha191')
os.chdir(BASE)

import engine
import evaluate

DATA = f'{BASE}/realdata'
OUT = f'{BASE}/results'
os.makedirs(OUT, exist_ok=True)
os.makedirs(f'{OUT}/ic', exist_ok=True)
os.makedirs(f'{OUT}/ls', exist_ok=True)

print('loading data...', flush=True)
daily = pd.read_parquet(f'{DATA}/csi500_daily.parquet', engine='fastparquet')
mem = pd.read_parquet(f'{DATA}/csi500_membership.parquet', engine='fastparquet')
idx = pd.read_parquet(f'{DATA}/csi500_index.parquet', engine='fastparquet')
info = pd.read_parquet(f'{DATA}/stock_info.parquet', engine='fastparquet')

print('building ctx...', flush=True)
ctx = engine.build_ctx(daily, idx[['date', 'open', 'close']])
dates, codes = ctx.C.index, ctx.C.columns
print('panel:', ctx.C.shape, flush=True)

mask = evaluate.membership_mask(mem, dates, codes)
print('mask日均成分数:', mask.sum(axis=1).mean().round(1), flush=True)

fwd = evaluate.forward_returns(ctx.C, (1, 2, 3, 5, 10))

# 行业宽表（常量按列广播）
ind_map = info.set_index('code')['industry']
industry = pd.DataFrame({c: ind_map.get(c, np.nan) for c in codes}, index=dates)

REB = pd.DataFrame({'date': dates})
YEARS = sorted(set(dates.year))

stats_rows = []
t_start = time.time()
for i in sorted(engine.REGISTRY):
    t0 = time.time()
    try:
        f = engine.REGISTRY[i](ctx).replace([np.inf, -np.inf], np.nan)
    except Exception as e:
        print(f'alpha{i:03d} COMPUTE FAIL: {e}', flush=True)
        continue
    row = {'factor': f'Alpha{i}'}
    EVAL_START = '2017-07-01'   # 报告样本外起点
    # --- 原始 RankIC, 多周期 ---
    for h in (1, 2, 3, 5, 10):
        ic = evaluate.rank_ic(f, fwd[h], mask)
        ic.name = f'alpha{i:03d}'
        ic.to_csv(f'{OUT}/ic/alpha{i:03d}_h{h}.csv')
        st = evaluate.ic_stats(ic.loc[EVAL_START:])
        row[f'RankIC{h}'] = st.get('IC均值')
        row[f'ICIR{h}'] = st.get('ICIR')
        row[f't{h}'] = st.get('t值')
    # --- 行业中性化 RankIC (h=1,5) ---
    try:
        f_neu = evaluate.neutralize(f, industry, pd.DataFrame(0.0, index=dates, columns=codes))
        for h in (1, 5):
            icn = evaluate.rank_ic(f_neu, fwd[h], mask)
            icn.name = f'alpha{i:03d}_neu'
            icn.to_csv(f'{OUT}/ic/alpha{i:03d}_neu_h{h}.csv')
            stn = evaluate.ic_stats(icn.loc[EVAL_START:])
            row[f'NeuRankIC{h}'] = stn.get('IC均值')
            row[f'NeuICIR{h}'] = stn.get('ICIR')
    except Exception as e:
        print(f'alpha{i:03d} neutralize fail: {e}', flush=True)
    # --- 十分位多空 (h=5 非重叠) ---
    try:
        bt = evaluate.decile_longshort(f, ctx.C, mask, hold=5)
        bt['group_returns'].to_csv(f'{OUT}/ls/alpha{i:03d}_groups.csv')
        ls = bt['ls']; ls.name = f'alpha{i:03d}'
        ls.to_csv(f'{OUT}/ls/alpha{i:03d}_ls.csv')
        stl = evaluate.ls_stats(ls, hold=5)
        row['LS年化'] = stl.get('年化多空收益')
        row['LS夏普'] = stl.get('夏普/IR')
        row['LS回撤'] = stl.get('最大回撤')
        row['LS胜率'] = stl.get('胜率')
    except Exception as e:
        print(f'alpha{i:03d} decile fail: {e}', flush=True)
    # --- 分年度 RankIC(h=5) ---
    ic5 = evaluate.rank_ic(f, fwd[5], mask)
    for y in YEARS:
        sub = ic5[str(y)]
        row[f'IC_{y}'] = sub.mean() if len(sub) > 10 else np.nan
    stats_rows.append(row)
    pd.DataFrame(stats_rows).to_csv(f'{OUT}/summary.csv', index=False)
    print(f"alpha{i:03d} done ({time.time()-t0:.1f}s) 累计{(time.time()-t_start)/60:.1f}min", flush=True)

print('ALL DONE', flush=True)
