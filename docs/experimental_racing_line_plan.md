# Experimental Racing-Line High-Level Planner

## 结论

可行，但建议作为实验分支做 A/B test，不要直接替换当前中心线 PPO 主线。

当前 high-level planner 能看到官方 5D：

```text
[lap_fraction, lateral_error_norm, boundary_margin_norm, heading_error_rad, curvature_norm]
```

这些信息足够支持一种“有限幅度走线”策略：在直道尽量回到中线，在弯道入口、弯中、出口根据 `lap_fraction` 和 `curvature_norm` 选择一个非零目标横向偏置，让机器狗不要一直贴着中心线跑。

这符合项目接口要求，前提是：

- actor 仍只消费官方 5D 观测。
- 输出仍是 `[vx, vy, yaw_rate]`。
- 不修改官方赛道几何。
- 目标路线始终在跑道边界内，不能为了走线越界。
- 报告中说明这是 high-level planner 的 learned/parameterized path bias，而不是额外观测。

## 为什么可行

`lateral_error_norm` 表示机器狗相对中心线的横向偏差，`boundary_margin_norm` 表示离边界还有多少安全余量。只要 planner 设定一个目标横向偏置 `target_lateral_norm(s)`，就可以把控制目标从：

```text
lateral_error_norm -> 0
```

改成：

```text
lateral_error_norm -> target_lateral_norm(lap_fraction)
```

其中 `target_lateral_norm` 可以是固定参数、少量可学习参数，或者 PPO policy 通过 reward 学出来的隐式策略。

对于固定 200m oval track，`lap_fraction` 已经告诉 planner 当前处在直道、弯道入口、弯中还是弯道出口。因此即使官方 5D 没有显式 lookahead，也可以根据固定赛道相位做分段走线。

## 合规性分析

项目要求 high-level planner 消费官方 5D 并输出 3D command。走线策略并不改变这个接口。它只是改变 policy 对 `lateral_error_norm` 的目标值。

需要注意的是，评分或 tie-break 可能考虑 lateral tracking error。如果我们偏离中心线太多，即使更快，也可能损失 line keeping 分数。因此实验中偏置应该保守：

```text
建议目标横向偏置: |target_lateral_norm| <= 0.35
对应实际偏置: |target_lateral_m| <= 0.70m
赛道半宽: 2.0m
```

这样仍保留约 1.3m 边界余量，风险较低。

## 走线直觉

对椭圆跑道，理想走线不是每时每刻贴中心线。更常见的策略是：

- 直道：回到中线附近，稳定加速。
- 入弯前：轻微靠外，为转弯留空间。
- 弯中：允许轻微靠内，减少有效转弯半径/转向负担。
- 出弯：逐渐回中，避免横向速度和 yaw rate 突变。

但 MuJoCo 里的 Go2 不是赛车，横向移动和大 yaw rate 都可能导致低层跟踪变差。因此偏置必须平滑，不能做激进 outside-inside-outside。

## 实验方案 A：Reward Shaping 走线

不改变 actor 输入结构，只修改 PPO reward。

### 目标线函数

定义一个保守目标偏置：

```text
target_lateral_norm = racing_line_bias(lap_fraction, curvature_norm)
```

初始版本可以用分段函数：

```text
直道: target = 0.0
弯道入口: target = -0.20 * turn_sign
弯中: target = +0.30 * turn_sign
弯道出口: target = 0.0
```

这里 `turn_sign` 用于区分左右转。如果只用当前 `curvature_norm`，它没有符号；但固定 oval 可以通过 `lap_fraction` 区分右弯和左弯。

### Reward 修改

当前 reward 有中心线惩罚：

```text
- lateral_weight * lateral_error_m^2
```

实验分支改成：

```text
- lateral_weight * (lateral_error_norm - target_lateral_norm)^2
- boundary_risk_weight * edge_risk
- smooth_line_weight * (target_lateral_norm_t - target_lateral_norm_{t-1})^2
```

保留边界惩罚，避免 policy 为了走线贴边。

### 优点

- actor 仍然完全合规。
- PPO 可以自己学习怎样用 `vy` 和 `yaw_rate` 贴近目标线。
- 走线失败时，只需要关掉 reward shaping。

### 风险

- 低层 checkpoint 可能不擅长非零 `vy`，走线会引入横向速度需求。
- 过大的偏置会增加 lateral tracking error，可能被评分扣分。
- 如果 heading 控制跟不上，入弯处可能更容易出界。

## 实验方案 B：Planner 内部目标偏置

在 high-level planner 内部显式计算目标偏置，并把它作为控制偏差：

```text
line_error = lateral_error_norm - target_lateral_norm
```

然后 planner 或 PPO action 后处理根据 `line_error` 调整：

```text
vy_bias = -k_line * line_error
yaw_bias = -k_heading_line * line_error
```

这可以和 PPO 输出相加，再经过现有 safety envelope。

### 合规写法

这个偏置只使用：

- `lap_fraction`
- `lateral_error_norm`
- `boundary_margin_norm`
- `curvature_norm`

所以仍然属于官方 5D 内部计算，不是额外状态。

### 推荐参数范围

```text
max_target_lateral_norm = 0.30 到 0.35
k_line = 0.05 到 0.12
max_extra_vy = 0.05 到 0.08 m/s
max_extra_yaw = 0.05 到 0.12 rad/s
```

先从很小的偏置开始，不要一上来追求明显 racing line。

## 实验方案 C：可学习走线参数

为了满足“learned planner or learned parameters”的要求，可以让 racing line bias 本身含少量可学习参数。

例如用 Fourier basis：

```text
target_lateral_norm(s) =
  a1 sin(2pi s) + b1 cos(2pi s)
  + a2 sin(4pi s) + b2 cos(4pi s)
```

或者分段 basis：

```text
target_lateral_norm = sum_i w_i * basis_i(lap_fraction)
```

训练方式：

1. PPO 训练 actor。
2. 同时让 `w_i` 作为 policy 参数一起学习，或用 CEM 搜索 `w_i`。
3. 强制 `target_lateral_norm = 0.35 * tanh(raw_target)`，防止越界。

这种方案报告起来比较干净：高层 planner 有 learned actor weights 和 learned racing-line parameters。

## 推荐实验顺序

1. 保留当前 PPO 主线作为 baseline。
2. 只改 reward，把 lateral error 目标从 0 改成小幅 `target_lateral_norm`。
3. 训练短版本：

```bash
python train_highlevel_ppo_torch.py \
  --checkpoint-dir best_checkpoint \
  --output-dir artifacts/highlevel_ppo_racing_line_debug \
  --total-updates 20 \
  --num-envs 64 \
  --rollout-steps 512 \
  --max-episode-seconds 240
```

4. 看日志：

```text
mean_progress_speed 是否上升
fall_count 是否暴涨
boundary_count 是否暴涨
mean_lateral_m 是否仍小于约 0.8m
```

5. 渲染 180 秒视频，对比中心线 PPO：

```bash
python run_track_bonus.py \
  --checkpoint-dir best_checkpoint \
  --planner-config artifacts/highlevel_ppo_racing_line_debug/planner_config.json \
  --output-dir artifacts/track_eval_racing_line \
  --duration-seconds 180 \
  --entry-name racing_line
```

6. 如果更快且不出界，再扩大训练；如果出界或变慢，回退。

## 成功标准

走线实验只有在以下条件同时满足时才值得保留：

- `valid_distance_m` 不下降。
- `mean_progress_speed` 上升。
- `boundary_violation = False`。
- `fall = False` 或 fall 不比 baseline 更差。
- `max_lateral_error` 不接近 2.0m。
- 视频中弯道轨迹平滑，没有蛇形摆动。

## 失败时的判断

如果走线导致更慢，可能不是 high-level 思路错，而是低层 policy 不擅长横向命令和弯道 yaw tracking。此时更保守的策略反而更好：

- 回到中心线。
- 只在入弯前提前降速。
- 减小 lateral target。
- 不使用 `vy`，主要用 `yaw_rate` 控制。

因此 racing-line 是值得试的实验方向，但不应该作为唯一最终方案。

