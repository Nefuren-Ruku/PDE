# 模型、数据与训练

**​<span style="color: orange;">Q：</span>​**: Task 2 可以使用 Task 1 的 checkpoint 吗？

**​<span style="color: orange;">A：</span>​**: ❌ 不可以。Task 2 需从头训练，不得使用 Task 1 的数据或 checkpoint。

**​<span style="color: orange;">Q：</span>​**: 可以使用预训练模型吗？

**​<span style="color: orange;">A：</span>​**:  Task 1可以使用 PDEBench 官方提供的 checkpoint 进行微调；Task 2需从头训练。

**​<span style="color: orange;">Q：</span>​**: 可以自己生成额外的训练数据吗？

**​<span style="color: orange;">A：</span>​**:  不可以。不允许使用数值求解器生成额外训练数据。只能使用官方提供的数据。

# 提交格式

**​<span style="color: orange;">Q：</span>​**: logs.log 中需要记录时间吗？

**​<span style="color: orange;">A：</span>​**: ✅ 需要。logs.log 中应记录 Agent 思考时间与模型训练时间的 wall clock time（实际耗时），以便核验 time.csv 中填写的时长是否准确。建议参考格式：

[2026-04-26 10:00:00] Agent 开始分析任务...
[2026-04-26 10:15:00] 开始训练 Epoch 1...

# 赛制相关

**​<span style="color: orange;">Q：</span>​**我可以一个人参加多个赛道吗？每个不同的赛道另建新队伍。

**​<span style="color: orange;">A：</span>​**可以。规则中“每人只能加入一支队伍”是指在同一赛道内。不同赛道允许您与不同的队友分别组队。

**​<span style="color: orange;">Q：</span>​**在 CNS 挑战赛中，我可以一个人参加多个任务吗？

**​<span style="color: orange;">A：</span>​**可以一个人参加多个任务。但请注意，CNS挑战赛作为一个赛道，其下不同任务应由同一支团队参与，不可出现“任务1与队友A组队、任务2与队友B组队”的情况。

# 赛题相关

**​<span style="color: orange;">Q：</span>​**如何均衡不同的gpu机器的时间效率问题？

**​<span style="color: orange;">A：</span>​** 本赛事以 **A100 (80GB)** 为标准 GPU。若选手使用其他 GPU 进行训练/推理，需将实测时间乘以对应折算系数，换算为 A100 等效时间后再计算得分。

**折算公式**：`A100等效时间 = 实测时间 × 折算系数`

- **H100 (80GB)**：2.0（性能约为 A100 的 2 倍，时间需乘 2）
- **H800 (80GB)**：1.8
- **A100 (80GB)**：1.0（基准）
- **A100 (40GB)**：1.0
- **A800 (80GB)**：0.9
- **RTX PRO 6000**：0.85
- **V100 (32GB)**：0.45
- **RTX 5090**：0.75
- **RTX 4090**：0.7
- **RTX 4080**：0.5
- **RTX 3090**：0.45
- **RTX 3080**：0.3
- **A10**：0.3
- **T4**：0.15
- **RTX 5060Ti**：0.1

> 例如：使用 V100 训练耗时 200 分钟，A100 等效时间 = 200 × 0.45 = 90 分钟。
> 使用 H100 训练耗时 30 分钟，A100 等效时间 = 30 × 2.0 = 60 分钟。

如使用上述未列出的 GPU，请联系组委会，组委会将酌情处理。

---

