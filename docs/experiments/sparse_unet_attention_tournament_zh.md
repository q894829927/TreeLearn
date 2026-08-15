# TreeLearn 多模型 Sparse U-Net 注意力竞赛

## 1. 目的与冻结规则

本分支在 TreeLearn 七层 Sparse Residual U-Net 内比较四类注意力机制，并保留普通局部微调与参数匹配非注意力控制。模型选择只使用固定训练集和五个验证森林，不读取 Wytham 标签。

- 分支：sparse-attention-unet
- 训练森林：A1N、A1W、G1N、G1W、G2N、G2W、G3N、G3W、L2N、L2W、LG1、LG2、LG3
- 验证森林：G4N、G4W、L1N、O1N、O1W
- L1W：胜者安全检查，不参与排名
- Wytham：最终一次锁定评价，不允许回头调参
- 固定种子：42、43、44
- grouping、seed keep ratio、HDBSCAN、semantic/base-offset 标签保持官方设置

如果某阶段 Gate 失败，记录失败并停止对应路线；不得选择“最不差”模型进入 Wytham。

## 1.1 T1 运行记录（2026-08-15）

- B1 Partial Fine-tune：训练完成，best epoch 7，峰值显存 5.666 GB。
- M1 Sparse-SE：混合精度池化修复后训练完成，峰值显存 7.131 GB。
- M2 Selective-Kernel：在统一 crop、batch size 1 和 RTX 4090 24 GB 上首次 backward 发生真实 CUDA OOM；当时 GPU 无其他计算进程。依据 T0/T1 硬件 Gate 淘汰，不为该模型单独缩小 crop，也不进入验证矩阵。
- M3 HCAG、M4 Window Attention：等待按相同训练设置继续。

`t1_validation_matrix.yaml` 使用 `enabled: false` 显式保留 M2 的淘汰阶段和原因。验证工具会写出 `eliminated_models.json`，不会把硬件失败伪装成缺失结果。

## 2. 已实现模块

统一入口位于 TreeLearn UBlock，支持 identity、residual_adapter、sparse_se、selective_kernel、hcag 和 window_attention。

所有模块均采用零初始化 gamma：

\[
F_{out}=F_{official}+\gamma\Delta F
\]

gamma 为 0 时输出严格退化为原始 UBlock。旧 YAML 未启用增强时不创建新增模块，旧 checkpoint 可按非严格模式直接载入。

统一微调范围：

- 新增 enhancement
- decoder level 0、1 的 deconv 和 blocks_tail
- semantic head
- offset head

input convolution、encoder 和其他层冻结；冻结 BatchNorm 始终保持 eval。

## 3. 关键文件

- tree_learn/model/unet_enhancement.py：M1–M4 与 B2
- tree_learn/model/blocks.py：UBlock 可配置接入
- tree_learn/model/tree_learn.py：冻结策略和 teacher preservation
- tools/training/train.py：梯度累积、分组学习率、早停、best_unet checkpoint
- tools/diagnostics/smoke_sparse_unet_tournament.py：T0 结构/显存/梯度 smoke
- tools/diagnostics/smoke_sparse_unet_pipeline.py：单 tile 端到端 smoke
- tools/diagnostics/run_sparse_unet_validation_matrix.py：五森林矩阵
- tools/diagnostics/summarize_sparse_unet_tournament.py：固定 T1/T2/T3 Gate
- tools/diagnostics/prepare_sparse_unet_adapter_control.py：自动参数匹配 B2
- tools/diagnostics/prepare_sparse_unet_preservation.py：生成 W-P 配置
- tools/diagnostics/prepare_sparse_unet_locked_evaluation.py：锁定 L1W/Wytham

## 4. 服务器同步与测试

~~~bash
cd ~/projects/zrx/code/TreeLearn
git fetch origin
git switch sparse-attention-unet
git pull --ff-only origin sparse-attention-unet

conda activate TreeLearn
mkdir -p logs/sparse_attention_unet

python -m unittest \
  tests.test_unet_enhancement \
  tests.test_unet_enhancement_spconv \
  tests.test_sparse_unet_tournament_config -v
~~~

测试失败时不要训练。

## 5. T0：结构、身份和显存验证

~~~bash
python -u tools/diagnostics/smoke_sparse_unet_tournament.py \
  --output_dir logs/sparse_attention_unet/t0_smoke \
  --num_points 4096 \
  --repeats 3 \
  --memory_limit_gb 23 \
  2>&1 | tee logs/sparse_attention_unet/t0_smoke_run.log

cat logs/sparse_attention_unet/t0_smoke/summary.md
~~~

运行五个单 tile 端到端 smoke：

~~~bash
set -o pipefail
for model in \
  b1_partial \
  m1_sparse_se \
  m2_selective_kernel \
  m3_hcag \
  m4_window_attention
do
  python -u tools/diagnostics/smoke_sparse_unet_pipeline.py \
    --config configs/experiments/sparse_attention_unet/t0_pipeline_\${model}.yaml \
    --output logs/sparse_attention_unet/t0_pipeline_\${model}.npz \
    2>&1 | tee logs/sparse_attention_unet/t0_pipeline_\${model}.log
done
~~~

确认 identity error 为 0、indices/metadata 不变、FP16 无 NaN、增强参数有梯度、峰值低于 23 GB，且单 tile 完成 prediction、ensemble、clustering 和保存。

如果 B1 完整训练 OOM，统一更换共享训练 crop；禁止只缩小某一个模型。

## 6. T1：seed 42 单种子初筛

~~~bash
nohup bash -c '
set -euo pipefail
for config in \
  train_t1_b1_partial_seed42 \
  train_t1_m1_sparse_se_seed42 \
  train_t1_m2_selective_kernel_seed42 \
  train_t1_m3_hcag_seed42 \
  train_t1_m4_window_attention_seed42
do
  echo "===== START \${config} $(date) ====="
  python -u tools/training/train.py \
    --config configs/experiments/sparse_attention_unet/\${config}.yaml \
    > logs/sparse_attention_unet/\${config}.log 2>&1
  echo "===== DONE \${config} $(date) ====="
done
' > logs/sparse_attention_unet/t1_training_runner.log 2>&1 < /dev/null &

echo $! | tee logs/sparse_attention_unet/t1_training_runner.pid
tail -f logs/sparse_attention_unet/t1_training_runner.log
~~~

随后跑五个验证森林并执行固定 Gate：

~~~bash
nohup python -u tools/diagnostics/run_sparse_unet_validation_matrix.py \
  --config configs/experiments/sparse_attention_unet/t1_validation_matrix.yaml \
  > logs/sparse_attention_unet/t1_validation_runner.log 2>&1 < /dev/null &

echo $! | tee logs/sparse_attention_unet/t1_validation_runner.pid

python -u tools/diagnostics/summarize_sparse_unet_tournament.py \
  --records logs/sparse_attention_unet/t1_validation/records.csv \
  --stage t1 \
  --baseline b1_partial \
  --output_dir logs/sparse_attention_unet/t1_summary \
  2>&1 | tee logs/sparse_attention_unet/t1_summary_run.log
~~~

T1 Gate：

- macro F1 相对 B1 至少 +0.2 pp
- Completeness 下降不超过 0.5 pp
- Commission 增加不超过 0.5 pp
- 至少 3/5 森林 F1 非负
- 最差森林不低于 -1.0 pp
- attention 统计有限且有非零方差

最多保留两个 finalist；无模型通过则关闭路线。

## 7. T2：三随机种子

T2 配置已为 B1、M1–M4 的 seeds 42/43/44 全部准备好，但只运行 B1 与 T1 finalists：

~~~text
configs/experiments/sparse_attention_unet/train_t2_<model>_seed<seed>.yaml
~~~

例如 finalist 为 M1 和 M3：

~~~bash
nohup bash -c '
set -euo pipefail
for model in b1_partial m1_sparse_se m3_hcag; do
  for seed in 42 43 44; do
    config=train_t2_\${model}_seed\${seed}
    python -u tools/training/train.py \
      --config configs/experiments/sparse_attention_unet/\${config}.yaml \
      > logs/sparse_attention_unet/\${config}.log 2>&1
  done
done
' > logs/sparse_attention_unet/t2_training_runner.log 2>&1 < /dev/null &
~~~

复制 T1 matrix 为 T2 manifest，仅保留 B1 与 finalists，并分别列出三个 seed。验证后运行：

~~~bash
python -u tools/diagnostics/summarize_sparse_unet_tournament.py \
  --records logs/sparse_attention_unet/t2_validation/records.csv \
  --stage t2 \
  --baseline b1_partial \
  --output_dir logs/sparse_attention_unet/t2_summary
~~~

T2 Gate 与胜者规则已固化。无通过模型则停止。

## 8. T3：参数匹配 B2

~~~bash
python -u tools/diagnostics/prepare_sparse_unet_adapter_control.py \
  --winner_config configs/experiments/sparse_attention_unet/train_t2_<winner>_seed42.yaml \
  --output configs/experiments/sparse_attention_unet/train_t3_b2_adapter_seed42.yaml

cat configs/experiments/sparse_attention_unet/train_t3_b2_adapter_seed42.parameter_match.json
~~~

只有参数差不超过 5% 才训练 B2。补 seed43/44、跑验证矩阵后执行：

~~~bash
python -u tools/diagnostics/summarize_sparse_unet_tournament.py \
  --records logs/sparse_attention_unet/t3_validation/records.csv \
  --stage t3 \
  --baseline b2_adapter \
  --winner <winner> \
  --output_dir logs/sparse_attention_unet/t3_summary
~~~

胜者必须比 B2 mean macro F1 高至少 0.3 pp，且至少 2/3 seeds 获胜。否则只能称为局部主干适配，不能称注意力有效。

## 9. T4：Seed-preservation

~~~bash
python -u tools/diagnostics/prepare_sparse_unet_preservation.py \
  --winner_config configs/experiments/sparse_attention_unet/train_t2_<winner>_seed42.yaml \
  --output configs/experiments/sparse_attention_unet/train_t4_winner_preservation_seed42.yaml \
  --weight 0.1
~~~

训练日志输出 teacher seed membership mismatch。只有固定验证集 macro F1、Completeness、Commission 均不恶化且 mismatch 至少降低 30%，才补 seeds 43/44。

## 10. 锁定 L1W 与 Wytham

~~~bash
python -u tools/diagnostics/prepare_sparse_unet_locked_evaluation.py \
  --winner_config <locked-winner-config> \
  --checkpoint <locked-best_unet.pth> \
  --tag <winner-tag>
~~~

先运行 L1W：

~~~bash
python -u tools/pipeline/pipeline.py \
  --config configs/experiments/sparse_attention_unet/locked/pipeline_l1w_winner_locked.yaml \
  2>&1 | tee logs/sparse_attention_unet/pipeline_l1w_winner_locked.log

python -u tools/evaluation/evaluate.py \
  --config configs/experiments/sparse_attention_unet/locked/evaluate_l1w_winner_locked.yaml \
  2>&1 | tee logs/sparse_attention_unet/evaluate_l1w_winner_locked.log
~~~

L1W Gate：F1 相对 98.4% 下降不超过 0.5 pp、Completeness 为 100%、Coverage 下降不超过 0.3 pp。

通过后 Wytham 只运行一次：

~~~bash
python -u tools/pipeline/pipeline.py \
  --config configs/experiments/sparse_attention_unet/locked/pipeline_wytham_winner_locked.yaml \
  2>&1 | tee logs/sparse_attention_unet/pipeline_wytham_winner_locked.log

python -u tools/evaluation/evaluate.py \
  --config configs/experiments/sparse_attention_unet/locked/evaluate_wytham_winner_locked.yaml \
  2>&1 | tee logs/sparse_attention_unet/evaluate_wytham_winner_locked.log
~~~

Wytham 后禁止更换模型、checkpoint、层级、窗口、teacher loss、seed 或 HDBSCAN。

## 11. 最终论文表

完整报告 B0、B1、B2、M1、M2、M3、M4、W、W-P 和已有 MLP confidence seed filtering。失败模型照实保留。只有通过 T1、T2、T3 和 L1W Gate 的模型可以作为“注意力有效”的论文主模型。
