# 数据字典模板

## 样本唯一标识

建议格式：

```text
YYYYMMDD-HHMM-{J|C序号}-R重复号
```

示例：`20260820-0930-C03-R1`。

## 核心字段

| 字段名 | 含义 | 推荐单位/格式 | 来源 | 备注 |
| --- | --- | --- | --- | --- |
| `sample_id` | 唯一样本编号 | 文本 | 原始标识 | 不得重复 |
| `date` | 采样日期 | YYYY-MM-DD | 原始记录 | 跨日期分组依据 |
| `time` | 采样时刻 | HH:MM | 原始记录 | 使用统一时区 |
| `sample_type` | 进出水标记 | J/C | 原始记录 | J=进水，C=出水 |
| `sample_no` | 出水序号 | C1–C5 等 | 原始记录 | 进水可留空或记 J |
| `replicate` | 重复号 | R1、R2… | 原始记录 | 区分技术/生物重复 |
| `aeration_l_min` | 曝气量 | L/min | 原始测量 | 记录仪表和设定值来源 |
| `temperature_c` | 温度 | ℃ | 原始测量 |  |
| `do_mg_l` | 溶解氧 | mg/L | 原始测量 | 记录测点与时刻 |
| `orp_mv` | 氧化还原电位 | mV | 原始测量 |  |
| `tn_in_mg_l` | 进水 TN | mg/L | 原始测量 |  |
| `tn_out_mg_l` | 出水 TN | mg/L | 独立实测 | 不用推导值替代 |
| `nh4_n_mg_l` | 氨氮 | mg/L | 原始测量 | 同时保留 J/C |
| `no3_n_mg_l` | 硝氮 | mg/L | 原始测量 | 低于定量限记 `<LOQ` |
| `no2_n_mg_l` | 亚硝氮 | mg/L | 原始测量 | 低于定量限记 `<LOQ` |
| `cod_mg_l` | COD | mg/L | 原始测量 | 同时保留 J/C |
| `snd_rate` | SND 率 | % 或 0–1 | 派生变量 | 固定公式与量纲 |
| `our_het_0` | 异养菌原始 OUR | mgO₂/(L·h) | 原始测量 |  |
| `our_het_max` | 异养菌最大 OUR | mgO₂/(L·h) | 原始测量 |  |
| `our_aob_max` | AOB 最大 OUR | mgO₂/(L·h) | 原始/计算 | 注明是否扣除基线 |
| `our_nob_max` | NOB 最大 OUR | mgO₂/(L·h) | 原始/计算 | 注明是否扣除基线 |
| `our_het_realtime` | 异养菌实时 OUR | mgO₂/(L·h) | 原始测量 |  |
| `our_aob_realtime` | AOB 实时 OUR | mgO₂/(L·h) | 原始测量 |  |
| `our_nob_realtime` | NOB 实时 OUR | mgO₂/(L·h) | 原始测量 |  |
| `atp_total` | 总 ATP | 检测方法规定单位 | 原始测量 | 与游离 ATP 同批次、同单位 |
| `atp_free` | 游离 ATP | 检测方法规定单位 | 原始测量 |  |
| `atp_intracellular` | 胞内 ATP | 同上 | 派生变量 | 总 ATP − 游离 ATP |

## 质量控制规则

1. 每个字段只使用一种标准单位，单位变化必须在导入阶段转换并留痕。
2. `<LOQ`、缺失、未检测和真实零值必须使用不同编码。
3. 原始值只读保存；清洗值与派生变量另建字段。
4. 每次模型运行记录数据版本、日期分组、特征清单和随机种子。
5. 同一天的全部样本必须进入同一训练或测试分组。

