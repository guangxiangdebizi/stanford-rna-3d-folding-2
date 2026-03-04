# Stanford RNA 3D Folding Part 2 — 比赛完整说明文档

> 数据来源：Kaggle 比赛页面（https://www.kaggle.com/competitions/stanford-rna-3d-folding-2）  
> 文档编写时间：2026-03-04

---

## 一、比赛基本信息

| 项目 | 内容 |
|------|------|
| 主办方 | Stanford University |
| 比赛类型 | Featured Code Competition（代码竞赛） |
| 奖金池 | $75,000 |
| 第1名 | $50,000 |
| 第2名 | $15,000 |
| 第3名 | $10,000 |
| 开始日期 | 2026-01-07 |
| 参赛截止 | 2026-03-18 |
| 提交截止 | 2026-03-25 |
| 私榜发布 | 2026-03-30（当周） |
| 当前参赛人数 | 约 1,817 人，1,703 支队伍 |

---

## 二、任务定义

### 核心问题

给定一段 RNA 分子的**核苷酸序列**，预测该 RNA 在三维空间中的**折叠结构**——即每个残基（nucleotide）上 **C1' 原子**的 3D 坐标（单位：埃 Å）。

### 输入

- `test_sequences.csv`：每行是一个 RNA 分子（target），包含：
  - `target_id`：唯一标识符
  - `sequence`：RNA 核苷酸序列（A/C/G/U，可能含修饰碱基或多链拼接）
  - `temporal_cutoff`：序列发布日期（yyyy-mm-dd）
  - `description`：来源描述（PDB 条目标题等）
  - `stoichiometry`：链的组成信息（形如 `{chain:count}`）
  - `all_sequences`：FASTA 格式的所有链序列（包含蛋白质、RNA 等多种成分）
  - `ligand_ids`：小分子配体的 PDB 三字母代码（`;` 分隔）
  - `ligand_SMILES`：配体 SMILES 字符串（`;` 分隔）

### 输出

对测试集中每个 `target_id`，预测 **5 套** 3D 结构，提交为 `submission.csv`。

---

## 三、提交文件格式

文件名必须为 `submission.csv`，格式与 `train_labels.csv` 相同，但包含 5 套坐标列：

| 列名 | 类型 | 含义 |
|------|------|------|
| `ID` | string | `{target_id}_{resid}`，residue 编号从 1 开始 |
| `resname` | char | 碱基字母（A/C/G/U） |
| `resid` | int | 残基序号（1-indexed） |
| `x_1, y_1, z_1` | float | 第 1 套结构中 C1' 原子坐标（Å） |
| `x_2, y_2, z_2` | float | 第 2 套结构坐标 |
| `x_3, y_3, z_3` | float | 第 3 套结构坐标 |
| `x_4, y_4, z_4` | float | 第 4 套结构坐标 |
| `x_5, y_5, z_5` | float | 第 5 套结构坐标 |

**注意：**
- `chain` 和 `copy` 列可不提供
- 坐标会被强制裁剪至 `-999.999` ~ `9999.999`（legacy PDB 格式限制）
- **必须给出 5 套坐标，缺任何一套会导致提交失败**

---

## 四、评估指标

### TM-score（Template Modeling Score）

$$
\text{TM-score} = \max\left(\frac{1}{L_{\text{ref}}} \sum_{i=1}^{L_{\text{align}}} \frac{1}{1 + \left(\frac{d_i}{d_0}\right)^2}\right)
$$

| 符号 | 含义 |
|------|------|
| $L_{\text{ref}}$ | 实验参考结构中被解析的残基数（ground truth 长度） |
| $L_{\text{align}}$ | 对齐的残基数 |
| $d_i$ | 第 $i$ 对对齐残基之间的距离（Å） |
| $d_0$ | 长度依赖的距离归一化因子 |

#### $d_0$ 的计算方式

$$
d_0 = 0.6 \cdot (L_{\text{ref}} - 0.5)^{1/2} - 2.5 \quad (L_{\text{ref}} \geq 30)
$$

| $L_{\text{ref}}$ | $d_0$ |
|---|---|
| < 12 | 0.3 |
| 12–15 | 0.4 |
| 16–19 | 0.5 |
| 20–23 | 0.6 |
| 24–29 | 0.7 |
| ≥ 30 | 按公式计算 |

#### 最终得分计算

- 对每个 target，取 5 套预测中 **TM-score 最高**的那套（best-of-5）
- 最终提交分数 = 所有 target 的 best-of-5 TM-score **平均值**
- TM-score 范围：0.0 ~ 1.0，**越高越好**
- 结构对齐使用 **US-align** 工具执行旋转+平移配准
- 对齐时只奖励高精度匹配（较大偏差残基对得分贡献极低）
- 若同一 target 有多个参考结构，取最优匹配

---

## 五、数据文件说明

### 数据规模

- 总文件数：约 15,319 个
- 总体积：310.63 GB
- 文件类型：`.cif`, `.fasta`, `.csv` 等

### 主要文件列表

```
stanford-rna-3d-folding-2/
├── train_sequences.csv          # 训练序列（含元数据）
├── train_labels.csv             # 训练标签（含实验 3D 坐标）
├── validation_sequences.csv     # 验证序列（=公开 test_sequences）
├── validation_labels.csv        # 验证标签（用于本地评估）
├── test_sequences.csv           # 真正的测试序列（无标签）
├── sample_submission.csv        # 提交格式示例
├── MSA/                         # 多序列比对（Multiple Sequence Alignment）
│   └── {target_id}.MSA.fasta   # 每个 target 的 MSA 文件
├── PDB_RNA/                     # Protein Data Bank RNA 结构库
│   └── {PDB_id}.cif             # 所有 RNA 相关 PDB 条目
│   └── pdb_seqres_NA.fasta      # 所有核酸链序列
│   └── pdb_release_dates_NA.csv # PDB 条目发布日期
└── extra/
    ├── parse_fasta_py.py        # 工具脚本：解析 all_sequences 字段
    ├── rna_metadata.csv         # 2025-12-17 前所有 RNA 结构的元数据
    └── README.md                # rna_metadata.csv 的字段说明
```

### train_labels / validation_labels 字段说明

| 列名 | 类型 | 含义 |
|------|------|------|
| `ID` | string | `{target_id}_{resid}` |
| `resname` | char | 碱基（A/C/G/U） |
| `resid` | int | 残基编号（1-indexed） |
| `x_1, y_1, z_1` | float | 第 1 套实验结构坐标 |
| `x_2, y_2, z_2` | float | 第 2 套（部分 target 有多个实验构象） |
| `chain` | string | 链 ID |
| `copy` | int | 链的拷贝编号（同一序列多个拷贝时 > 1） |

### 数据划分方式

- 使用 **MMseqs2 聚类 + 时序截断** 的方式划分 train/validation，避免 train 与 val 序列同源
- validation 进一步过滤：至少 40% RNA 组成、序列唯一性（<90% 相似）
- train 不做去冗余，但只包含通过质量筛选的 PDB 条目

### 数据质量筛选标准（train/validation 均适用）

- 残基类型：标准 ACGU 或可映射到标准碱基的修饰残基
- 无未定义（N）残基，无 T（DNA）残基
- 修饰残基比例 ≤ 25%
- 至少 50% 序列中的残基在结构中被观测到
- 所有 RNA 链的总结构化程度（structuredness）≥ 20%
- 总残基数 ≥ 10 nt

---

## 六、代码竞赛运行约束

| 约束项 | 限制 |
|--------|------|
| CPU Notebook 时限 | ≤ 8 小时 |
| GPU Notebook 时限 | ≤ 8 小时 |
| 网络访问 | 禁止（Internet disabled） |
| 外部数据 | 允许使用公开可获取的数据和预训练模型（需提前上传到 Kaggle Dataset 或 Model Hub） |
| 提交方式 | 必须通过 Notebook commit 提交（不可直接上传 csv） |
| 输出文件名 | 必须为 `submission.csv` |
| 运行时间抖动 | 重复相同提交，评分时间可能有 ±5 分钟偏差（正常现象） |

### 隐藏测试集（Future Data）评估

- 比赛结束后，主办方会使用最新解析的 RNA 结构（未在训练数据中出现）重新评分
- 隐藏数据测试阶段的时限将按样本数量比例扩展
- 你的代码必须能完整跑完（包括该阶段），否则最终成绩无效

---

## 七、奖励与额外权益

### 奖金

| 排名 | 奖金 |
|------|------|
| 第 1 名 | $50,000 |
| 第 2 名 | $15,000 |
| 第 3 名 | $10,000 |

### 论文署名

- 私榜前排选手将被邀请参与比赛总结论文的共同署名
- 需提交代码和模型描述

---

## 八、主要技术挑战

1. **3D 坐标回归任务**：不是分类，是对每个残基预测实数坐标，坐标空间复杂
2. **序列长度差异大**：RNA 序列短至十几 nt，长至数百 nt，模型需处理变长输入
3. **5 候选多样化**：需生成结构多样的 5 套预测，best-of-5 机制对结构多样性有要求
4. **推理效率限制**：8 小时内跑完全部测试集，不能用慢模型
5. **修饰碱基**：部分序列含非标准碱基（超出 ACGU），需要预处理
6. **多链结构**：部分 target 含多条 RNA 链拼接，链间交互复杂
7. **数据泄露控制**：测试 target 对应的 PDB 数据在训练截止后才发布，需用 temporal_cutoff 过滤

---

## 九、参考资料（官方提供）

- [2024 CASP16 挑战赛论文](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2）
- Part 1 比赛总结论文：*Template-based RNA structure prediction advanced through a blind code competition*
- [US-align 工具主页](https://zhanggroup.org/US-align/)
- [TM-score 评分代码（官方开放）](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/overview/evaluation)
