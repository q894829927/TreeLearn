# 局部拆分的下一步：先导出完整混合实例和正确对照

## 本轮做什么

上一次完整计数分析表明，任意两原实例并集只有 6 棵漏检达到 IoU > 0.5 的机会，
任意多实例并集也只有 7 棵；137 棵漏检已在单个混合实例里保留超过半棵树点。
下一步先验证完整局部点云是否具有能用于拆分的证据，不再启动关系 V3 训练。

新增只读 CPU 工具 `tools/instance_relation_v3/export_split_cases.py`，默认导出：

| 原实例 | 已知混合目标 GT | 目标点召回 |
|---|---|---|
| 1767 | 38、292 | 98.96%、99.18% |
| 250 | 176、424 | 96.74%、98.45% |
| 262 | 622、129 | 95.08%、99.59% |
| 1427 | 49、783 | 96.50%、94.02% |

这四个原实例在官方基线里都不是 TP。对每个混合实例，工具另选一个不重复的正确实例作对照：
正确实例的已知点主导 GT 纯度至少 0.8、已知点比例至少 0.5，并在点数、Z 跨度、XY 位置上匹配。
对照 ID 由服务器实际缓存决定，最终打印在 `CASE_PLAN` 和 `brief.json` 中。
这些选择都使用了 GT，只是事后诊断案例，不是部署候选选择器。

## 服务器前台执行

```bash
cd ~/projects/zrx/code/TreeLearn
conda activate TreeLearn
git pull --ff-only origin codex/candidate-mask-mlp-step1

mkdir -p logs/instance_split_cases
set -o pipefail
python -m unittest discover -s tests -p 'test_instance_split_case_export.py' -v &&
python -u tools/instance_relation_v3/export_split_cases.py \
  2>&1 | tee logs/instance_split_cases/export.log
```

终端会依次显示：

```text
CASE_PLAN [...4 个 mixed_case，4 个 detected_control...]
INDEX_CORE points=.../37264001
INDEX_CONTEXT points=.../37264001
SAVE original=... role=... complete_core=... context=...
UPLOAD=/.../split_casepack.zip bytes=...
```

不加载模型、torch、checkpoint 张量，不训练，不占 GPU，不修改缓存或预测。
工具读取现有官方缓存，两轮扫描分别收集完整实例索引和有界上下文索引，
再仅为入选点读取冻结特征和审计标签，不复制整场特征数组。
不需要停止正在运行的其他只读实验；若缓存被同时改写，导出会拒绝完成。

输出目录为：

```text
work_dirs/treelearn_instance_relation_v3_s42/split_case_analysis/
  export_<时间>_<随机后缀>/split_casepack.zip
```

**把最后一行 `UPLOAD=` 指向的 `split_casepack.zip` 上传回来。**
每次生成新目录，不覆盖先前案例，也不覆盖第 2000 步的 V3 结果。

## 包中内容

每个 `original_<ID>/` 下保存：

- `inputs.npz`：原始坐标 `coords`、官方 XY 投票 `votes_xy`、树概率 `tree_probability`、
  32 维冻结特征 `features`、官方原实例编号 `baseline`、全场点索引 `source_indices`、
  原目标实例掩码 `is_core`。其中不含 GT。
- `audit.npz`：与输入索引严格对齐的 GT `labels` 和 `source_indices`。
  GT < 0 是未知，GT 0 是已知背景，不把未知点改成负例。
- `case.json`：完整目标点数、上下文抽样量、完整 GT 计数复核、文件哈希，以及诊断角色。

根目录的 `brief.json` 记录基线指标、全部案例及来源文件戳；`reference_counts.npz` 保存紧凑精确计数。
如果之前导出的重建归一化统计存在，另附 `normalization.json`；原输入特征本身不做归一化变更。

### 完整目标与上下文的区别

目标原实例包含它原来拥有的**全部点**，包括未知／背景标签点，绝不只挑属于目标 GT 的点。
默认完整目标上限为每例 200,000 点；超过上限直接报错，不冒充完整导出。

上下文框由原实例完整点集定义，XYZ 各扩展 2m；上下文不含该原实例自己的点。
上下文按框内相对高度分 4 层，每层最多 2,048 点，共最多 8,192 点。
用全场点索引的确定性哈希抽样，扫描批次大小不会改变抽样集合。
**上下文有界抽样，不是完整周边森林，也不是地形归一化高度。**

## 上传后判断什么

1. 混合实例是否存在两条独立下部轨迹、跨层支撑或可分支干；正确实例中会不会出现类似的多分支形态。
2. 官方投票是否有多个可重复的局部峰，还是不同树的票已经重叠。
3. 原始几何、冻结特征、局部连接分别能否支持点归属；不能用二维 PCA 混合直接认定特征坍缩。
4. 如果局部拆分可行，再设计仅编辑选中原实例的小分支；未选中实例点集保持不变。
   全林仍统计 lost/recovered、FP/FN 和分割质量，不能只展示挑选的成功案例。

本轮不训练拆分网络，不产生新模型成绩，也不承诺 137 棵理想机会都能实现。
Wytham 案例已参与设计诊断，后续提升应作为探索结果，需用其他保留森林检验泛化。

## 校验范围

新增四项服务器 CPU 测试，覆盖完整点集、索引对齐、上下文预算与扫描批次不变性、
GT 单独保存且在输入固定后读取、未知标签保留、计数变化拒绝、正确实例保护及输出目录隔离。
本地只做语法和静态检查；测试由上述服务器命令执行。
没有修改训练模块、V3 签名输入、模型配置或旧结果。
