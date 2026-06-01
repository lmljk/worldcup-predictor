---
name: worldcup
description: |
  2026 世界杯(美/加/墨, 48 队 104 场)逐场预测。市场锚定集成模型:去抽水市场共识赔率为先验,
  叠加 Dixon-Coles 双变量 Poisson + ELO + 情境因子(海拔/天气/休息/伤停)做偏移与 edge 检测;
  全局蒙特卡洛模拟出小组出线/晋级/夺冠概率。赛中赛后按实时赔率/Polymarket/阵容每日复盘调整。
  数据底座复用 machina-sports/sports-skills(football-data/polymarket/betting)。
  触发词: "/worldcup" / "跑一下 worldcup" / "world cup 预测" / "世界杯预测" / "今天世界杯" /
  "预测这场" / "夺冠概率" / "复盘世界杯" / "worldcup 看板".
allowed-tools:
  - Bash(/Users/bot/worldcup/.venv/bin/python:*)
  - Bash(python3.11:*)
  - Read
  - Write
  - Edit
  - Glob
  - Grep
  - WebSearch
  - WebFetch
---

# /worldcup — 2026 世界杯逐场预测

## 这个 skill 做什么

对 2026 世界杯每一场比赛给出校准过的概率预测(1X2 胜平负 / 比分分布 / 大小球 / 双方进球),
并用蒙特卡洛模拟整届赛制,输出每队小组出线、各轮晋级、夺冠概率。赛前给预测,赛中赛后按
实时数据(博彩赔率、Polymarket、伤停、确认首发、天气)每日复盘调整,结果挂到公开看板
`worldcup.polyalpha.cn`。

## 何时触发

- 用户说"跑一下 worldcup / 预测这场 / 世界杯预测 / 夺冠概率 / 复盘世界杯"。
- 赛事期间每日定时复盘(launchd)。

## 模型哲学:市场锚定集成

四层,每场输出校准后概率:
1. **市场先验** `P_market`:多家博彩 1X2 去抽水(de-vig)+ Polymarket 价格,按流动性加权。
2. **统计模型** `P_model`:Dixon-Coles 双变量 Poisson(低分修正 + 指数时间衰减 ξ)估每队 λ;ELO 差作先验。
3. **情境调整**:在 λ 上做小幅乘性调整(海拔/高温/休息天数/旅行/关键伤停),幅度由回测约束。
4. **集成+校准**:`P_final = w·P_market + (1-w)·P_model_adj`,w 由回测定;过 isotonic 校准。
   `edge = P_final − P_market` 是模型相对市场的分歧,看板专门展示。

## 跑命令

```bash
PY=/Users/bot/worldcup/.venv/bin/python   # 若无 venv 用 python3.11
cd /Users/bot/worldcup

# 数据刷新(历史/ELO/赛程/天气/实时赔率/polymarket)
PYTHONPATH=. $PY -m skill.helpers.cli fetch --all

# 单场预测
PYTHONPATH=. $PY -m skill.helpers.cli predict --match <fixture_id>

# 全部场次 + 蒙特卡洛晋级树
PYTHONPATH=. $PY -m skill.helpers.cli predict --all --simulate

# 模型 vs 市场(Polymarket 去抽水夺冠赔率 + edge)
PYTHONPATH=. $PY -m skill.helpers.cli market

# 回测(walk-forward, 防 lookahead)
PYTHONPATH=. $PY -m skill.helpers.cli backtest --start 2010-01-01 --end 2026-05-31

# 每日复盘(赛中)
PYTHONPATH=. $PY -m skill.helpers.cli review --date today
```

## 硬约束(本工作区投研铁律)

- 回测只用 ≤T 日数据,ELO/衰减 as-of 重算,严禁未来函数。
- 任何因子必须在 walk-forward 里相对 baseline 稳定正贡献才保留;禁止为凑分手调参数。
- 样本不足的因子明确标低置信,不假装有效。
- 开源脱敏:无个人邮箱/飞书 ID/本机路径/token/key 进库;key 走 .env。

## 数据底座

复用 machina-sports/sports-skills(已装):`football-data`(赛程/比分/xG/伤停)、
`polymarket`(免费盘口)、`betting`(de-vig/edge/Kelly)。免费数据源详见 README。
