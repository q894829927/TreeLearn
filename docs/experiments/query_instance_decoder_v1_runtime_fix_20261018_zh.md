# 查询解码器 v1：首次完整验证中断修复

日期：2026-10-18。依据用户回传的服务器日志，原提交 `2ac78b3` 的后台任务 `run_20261018T181615_HPi2kQ` 已退出，状态 `FAILED stage=train exit=1`，不是完整实验完成。

## 已经确定的原因

原 10 项合成测试、建索引、8 step smoke 训练和 smoke 预测均完成。正式训练到第 2000 step 后，开始首次五森林完整验证：G4N 已完成，G4W 经过第 175/393 块后，在推理中触发：

```text
RuntimeError: Too many dense proposal memberships in one block; inspect mask/score learning.
```

这是代码主动检查累计查询—点成员数超过 2000 万后抛出的异常，不是本次回传日志中的 CUDA OOM、Linux SIGKILL 或 loss 非有限错误。成员数是所有通过质量阈值查询的掩码成员总和，包含同一树/同一点被多个查询覆盖的重复量；超限本身不能证明每个掩码都异常膨胀。

训练日志第 2000 step 的 loss 为 0.5340，各损失分项有限。训练累计约 409 秒，记录峰值 RSS 3259953152 bytes、CUDA allocated 352343552 bytes；这两个数来自训练日志，不能当作完整验证的峰值。

旧实现先验证再保存该步的 checkpoint，导致此次第 2000 step 权重未保存；在默认每 500 step 的保存周期下，`last.pth` 应停在第 1500 step。最终 `evaluate` 阶段没有执行，尚无完整五森林汇总。

## 修复内容

1. 单块小掩码继续用内存缓存，超过原成员预算后将每个查询的完整成员序列写入块级临时文件，再以 mmap 供块内 NMS 读取。所有查询仍预测全部细点，不截断成员、删查询或提高质量阈值；同一分数、掩码和 NMS 参数应给出同样结果。临时目录在块结束或发生 Python 异常时清理。
2. 单块预算改为内存缓存阈值，保持 YAML 数值不变，以兼容已保存的训练配置和 checkpoint。新增 `block_masks_spilled` 日志、溢写块数及实际单块最大成员数。全林提案文件的磁盘预算保持有效。
3. 在完整验证前原子保存最新 `last.pth`，标记 `pending_validation=true`；完整验证成功后清除标记并更新 best。若验证失败，恢复时先以同一步权重重试验证，再继续训练。
4. 恢复启动器先执行当前 12 项服务器测试。新增磁盘/内存掩码存储与 NMS 一致性、验证失败保存与恢复测试，原 GT 隔离/分 chunk 测试也覆盖强制磁盘路径。本地只做 Python AST、Bash 语法、文档链接及 Git 静态检查，未执行这些测试或模型。

本次保持网络、训练目标、查询预算、模型/掩码阈值和 NMS 策略不变，修复运行与保存逻辑；不因 G4N 的单次指标改写本轮训练配置。

## 当前结果能够说明什么

第 2000 step 的 G4N 相对原版变化：

| 指标 | 变化 |
|---|---:|
| F1 | -19.524 pp |
| Completeness | +2.516 pp |
| Commission | +33.082 pp |
| Coverage | +1.846 pp |
| Point precision | +4.825 pp |
| Point recall | -3.209 pp |

这说明在这片森林、这个时点，实例匹配召回与平均最佳重叠有所提升，但误检比例明显增大，整体 F1 变差。不能称为有效提升。跨块重复、碎片预测、质量分数校准各占多少，还需结合最终实例数及完整评价判断；当前日志不足以分别归因。

G4N 共有 199 个推理块，块内 NMS 后留下 3997 个提案、118694696 个成员引用，随后才做全林 NMS。3997 不是最终树数，1.187 亿也不是森林独立点数。

## 在服务器恢复

```bash
cd ~/projects/zrx/code/TreeLearn
git pull --ff-only
conda activate TreeLearn
bash tools/query_segmentation/launch.sh resume
```

无需重新生成缓存或索引，也无需删除 checkpoint。原版 checkpoint 兼容，启动日志 `train_start` 的 `start` 将显示实际恢复位置；此次预期为 1500，重跑 500 step 后重新验证。

```bash
cat logs/query_instance_decoder_v1/latest/status.txt
tail -n 20 logs/query_instance_decoder_v1/latest/console.log
tail -n 20 logs/query_instance_decoder_v1/latest/train.log
```

`RUNNING tests` 时查看 `tests.log`；进入训练后，密集块可能出现 `block_masks_spilled`，表示正常的磁盘暂存。完整操作与最终结果路径见[运行说明](query_instance_decoder_v1_zh.md)。
