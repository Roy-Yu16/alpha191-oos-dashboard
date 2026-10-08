# Alpha191 因子计算代码包（国泰君安191 · 中证500 样本外检验）

对应研究报告：《Alpha191研究报告_样本外检验》。本包包含 191 个价量因子的完整计算实现与评价流水线。

## 目录结构（建议放入 D:\quant\191\）

```
D:\quant\191\
├── code\            ← 本压缩包 code\ 下的全部文件
│   ├── operators.py        算子库（RANK/DELTA/TSRANK/SMA/REGBETA/HIGHDAY 等）
│   ├── alphas_001_050.py   因子函数 1–50
│   ├── alphas_051_100.py   因子函数 51–100
│   ├── alphas_101_150.py   因子函数 101–150
│   ├── alphas_151_191.py   因子函数 151–191（含 143 递归、149 条件回归等）
│   ├── engine.py           数据装配（前复权拼接、VWAP、RET、基准广播）+ 因子注册表
│   ├── evaluate.py         评价模块（PIT成分掩码、前瞻收益、行业中性化、RankIC、十分位多空）
│   ├── run_pipeline.py     全量流水线入口（191因子 × 5 horizon IC + 中性化 + 多空 + 分年度）
│   └── formulas.json       191 条公式文档（来源逐条标注：报告OCR / 公开转录）
├── data\            ← 你自己的数据（本包不含）
│   ├── csi500_daily.parquet       日线面板（date, code, close, qfq_*, volume, amount, pct_chg, limit_*）
│   ├── csi500_membership.parquet  PIT 成分区间表（code, in_date, out_date）
│   ├── csi500_index.parquet       中证500 指数日线
│   └── stock_info.parquet         股票信息（含 industry 行业分类）
└── output\          ← 流水线输出（results\ic\、results\ls\、summary.csv）
```

## 运行

```bash
cd D:\quant\191\code
python run_pipeline.py
```

流水线内的数据/输出路径常量（`DATA_DIR`、`OUT_DIR`、`EVAL_START`）按上面目录结构修改即可。

## 依赖

Python 3.10+；pandas、numpy、bottleneck、fastparquet（或 pyarrow）、scipy。

## 关键口径（与研究报告一致）

- 因子在 t 日收盘计算，前瞻收益 t 收盘 → t+h 收盘；h ∈ {1,2,3,5,10}
- RET 用 `pct_change(fill_method=None)`，停牌不产生虚假收益
- TSRANK 基于 bottleneck `move_rank`（输出 [-1,1]），映射为 1..n
- SMA(x,n,m) = `ewm(alpha=m/n, adjust=False)` 递归平滑
- 成分掩码：`in_date <= t < out_date`，月度粒度 PIT
- 中性化：逐日截面回归行业哑变量取残差（规模因子待 v2 数据补充）
- 特殊实现标注：Alpha30 为 MKT-only 残差（SMB/HML 不可得）；Alpha51 为上移缺口占比；Alpha143 递归初值 1；Alpha149 用 252 日窗口内基准下跌日子样本（min_periods=60）

## 修改公式的入口

每个因子是独立函数 `alphaNNN(ctx)`，`ctx` 提供宽表（date × code）：O/H/L/C/V/AMOUNT/RET/VWAP 与基准 BC/BO/MKT。改公式只动对应 `alphas_*.py` 中的函数即可，注册表自动收集。
