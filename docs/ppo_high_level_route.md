# Go2 Oval Track 高层 PPO 技术路线

## 结论先行

可以用机器人自身速度、加速度、跌倒状态、能耗、足端滑移、与赛道边界/中心线的相对位置等信息来构造 PPO 的训练环境、奖励函数和终止条件；但最终提交的高层 actor 建议仍严格只消费官方 5D track observation：

```text
[lap_fraction, lateral_error_norm, boundary_margin_norm, heading_error_rad, curvature_norm]
    -> [vx_mps, vy_mps, yaw_rate_radps]
```

原因是任务规则明确要求 high-level planner 消费官方 5D 观测。仿真内部的 qpos/qvel、真实速度、加速度可以作为 reward shaping、reset curriculum、日志指标，甚至可以作为 critic 的辅助信息；但如果 actor 直接依赖官方接口没有提供的机器人速度/加速度，线下训练会和官方评测接口产生信息不一致风险。

本仓库新增的实现采用保守合规方案：

- actor 输入：官方 5D。
- actor 输出：连续高层命令 `[vx, vy, yaw_rate]`。
- reward/termination：使用完整仿真动态量，包括进度、横向误差、边界裕度、跌倒、速度变化、能耗和足端滑移。
- 低层 policy：保持 Assignment 1 / Brax PPO checkpoint 格式，不重新训练。
- 训练框架：PyTorch PPO 训练高层 actor-critic，JAX/MJX 负责固定低层 policy 和物理 rollout。
- 部署格式：导出 `planner_weights.npz` 和 `planner_config.json`，评测时 `track_bonus/planner.py` 用 NumPy 推理，不依赖 PyTorch。

## 目前代码的技术路线

当前项目的主结构是分层控制：

```text
官方 5D 赛道观测
  -> high-level planner
  -> [vx, vy, yaw_rate]
  -> HW1-style Go2 low-level Brax PPO policy
  -> 12 维关节动作
  -> MuJoCo/MJX 仿真
```

关键文件：

- `track_bonus/controller_interface.py`：定义官方 5D 观测和 3D 高层命令接口。
- `track_bonus/planner.py`：实现 starter PD planner 和已有 learned MLP planner。
- `run_track_bonus.py`：官方式单 policy rollout、打分、视频和结果导出。
- `track_bonus/scoring.py`：根据进度、完成圈、跌倒、出界、横向误差、能耗、滑移计算指标。
- `train_mlp_cem.py`：当前高层优化主要路线，用 CEM 搜索 MLP 权重。

当前 CEM 路线大致是：

1. 固定低层 Go2 PPO checkpoint。
2. 用 JAX/MJX 跑低层仿真。
3. 高层 MLP 输出 `[vx, vy, yaw_rate]`。
4. CEM 每代采样一批 MLP 参数，向量化 rollout，按 lap completion、速度、line keeping、稳定性打分。
5. 取 elite 更新均值和方差，保存最优 `planner_weights.npz`。

已有 CEM 版本有几个优点：

- 不需要对低层 policy 反传，和现有 JAX rollout 结合简单。
- 高层参数少，CEM 容易调试。
- 已经加入 command smoothing、速度上限、安全包络等工程保护。

主要限制：

- CEM 是黑盒优化，样本效率不高。
- 对随机初始状态、局部扰动、长时序 credit assignment 的利用较弱。
- 高层策略更新只看整段 rollout 的综合得分，难以精细利用每一步的进度、边界裕度和稳定性反馈。
- 当前 learned MLP 内部扩展到 8D 特征，包括估计速度和前方曲率；这在工程上有用，但报告和提交时需要解释为 planner 内部从官方 5D 历史和固定赛道几何派生的状态，而不是额外官方观测。

## 新技术路线：PyTorch PPO 训练高层 policy

新路线把高层 planner 当作一个小型连续控制 RL policy：

```text
obs_5d_t
  -> PyTorch actor MLP
  -> raw Gaussian action
  -> tanh/sigmoid bounded command
  -> safety envelope + command delta limit
  -> fixed low-level Go2 policy
  -> MJX step
  -> reward_t, done_t, obs_5d_{t+1}
```

### 1. Observation 设计

提交 actor 使用官方 5D：

- `lap_fraction`：当前圈内进度。
- `lateral_error_norm`：相对中心线横向偏差，按半赛道宽归一化。
- `boundary_margin_norm`：离边界的剩余安全距离，按半赛道宽归一化。
- `heading_error_rad`：机器人朝向与赛道切线方向误差。
- `curvature_norm`：当前位置曲率，直道为 0，弯道约为 1。

训练环境内部额外计算但不直接喂给 actor：

- progress speed：由连续 `s` 差分得到。
- acceleration proxy：progress speed 的差分。
- base height / done：用于跌倒判断。
- actuator force 和 joint velocity：能耗 proxy。
- foot velocity：足端滑移 proxy。
- cumulative progress：判断是否完成一圈。

如果后续要更激进，可以加入两种合规增强：

- frame stacking：把最近几帧 5D 拼起来，速度信息来自官方观测历史。
- recurrent planner：用 LSTM/GRU 在 planner 内部维护状态，但评测入口仍每步只接收 5D。

### 2. Action 设计

PPO actor 输出 3 维 raw action，训练时按如下方式变成低层命令：

```text
vx       = 0.5 * max_vx * (tanh(raw[0]) + 1)
vy       = max_vy * tanh(raw[1])
yaw_rate = max_yaw_rate * tanh(raw[2])
```

默认范围：

- `max_vx = 1.0 m/s`
- `max_vy = 0.22 m/s`
- `max_yaw_rate = 0.60 rad/s`

动作后处理：

- 前 `stand_seconds` 输出零命令，避免刚 reset 就摔。
- 用 safety envelope 根据 heading/lateral/boundary/curvature 风险降低速度上限。
- 用 `max_command_delta` 限制相邻步命令变化，减少低层 tracking 突变。

### 3. Reward 设计

核心目标是 dense reward + 官方指标一致：

- 正奖励：每步有效赛道进度 `delta_s`。
- 小正奖励：保持边界裕度。
- 惩罚：横向误差、heading error、命令变化过大、速度突变、能耗、足端滑移。
- 终止惩罚：跌倒、出界。
- 完圈奖励：累计进度达到 200m。

实现中的 reward 形状：

```text
reward =
  progress_reward_scale * delta_s
  + speed_reward_scale * clipped_progress_speed
  + target_speed_reward_scale * clipped(progress_speed / target_speed)
  - slow_penalty_scale * below_target_speed_gap
  - backward_penalty_scale * backward_speed
  + 0.02 * clipped_boundary_margin_norm
  - 0.35 * lateral_error_m^2
  - 0.08 * heading_error_rad^2
  - 0.025 * ||command_t - command_{t-1}||^2
  - 0.00002 * acceleration_proxy^2
  - 0.0005 * energy_proxy
  - 0.015 * foot_slip_proxy
  - boundary_penalty_if_outside
  - fall_penalty_if_fallen
  + finish_bonus_if_full_lap
```

这些系数不是最终最优解，而是一个合理起点。建议训练时记录 `training_history.json`，根据失败模式调：

- 总是冲出弯道：增大 lateral/heading/boundary 惩罚，降低 `max_vx` 或增大 edge slowdown。
- 直道太慢：增大 progress reward 或 `max_vx`。
- 原地抖动：增加 progress speed 奖励，减小 command delta 惩罚。
- 容易摔：降低 `max_vx`、增加 stand time、提高动作平滑。

### 4. Curriculum / reset

默认 `--start-randomization curriculum`：

- 训练初期更多从 `start_s_m=0` 出发，先学直道稳定起跑。
- 随 update 增加全赛道随机 reset 比例，让策略覆盖弯道入口、弯中、弯道出口和上直道。
- reset 时加入小 lateral/heading perturbation，提高鲁棒性。

可选：

- `fixed`：只从指定 `start_s_m` 训练，适合调试。
- `full_track`：全赛道随机 reset，适合已有稳定策略后微调。

### 5. PPO 算法设置

默认配置：

- `num_envs = 64`
- `rollout_steps = 512`
- `total_updates = 120`
- `max_episode_seconds = 240`
- `max_vx = 1.25`
- `target_straight_speed = 1.10`
- `target_curve_speed = 0.70`
- `gamma = 0.995`
- `gae_lambda = 0.95`
- `clip_eps = 0.20`
- `ppo_epochs = 4`
- `minibatch_size = 1024`
- actor/critic MLP：2 层 hidden，每层 64，Tanh。

高层任务的有效 horizon 很长。助教演示一圈约 2 分 46 秒，即 166 秒；因此训练 episode 上限必须高于这个时间。`rollout_steps` 不需要一次覆盖完整一圈，因为并行环境会跨 PPO update 延续状态，但 `max_episode_seconds` 如果低于一圈时间，策略永远看不到真正的 full-lap terminal/bonus。当前推荐用 240 秒 episode 上限，并用 180 秒或 300 秒评估视频检查完整跑圈。

## 可行性分析

整体可行，原因：

- 高层 action 只有 3 维，比直接训 12 维关节 policy 简单很多。
- 低层 policy 已能走直线，高层 PPO 只需要学习何时减速、转向、纠偏。
- 官方 5D 已经包含赛道相对位置和曲率，足够表达路线跟踪任务的主要状态。
- reward 可以利用完整仿真状态，PPO 比 CEM 更能使用每一步 dense signal。

主要风险：

- 低层 checkpoint 的 yaw/side velocity tracking 可能很弱。高层再聪明也不能让低层执行超出训练分布的命令。
- 官方 5D 没有显式速度，actor 对动态状态部分可观测不足。速度只能通过策略行为和观测变化间接推断。
- PPO 与 MJX/JAX 环境桥接有 CPU/GPU 数据拷贝开销，训练速度可能不如纯 JAX CEM。
- Reward shaping 不当会导致策略“保守慢跑”或“短期冲刺后出界”。
- 部署中的 safety envelope 会改变 actor 原始输出，因此训练和部署应保持一致。

风险缓解：

- 默认 actor 只输出低层较可能跟踪的安全命令范围。
- 训练 rollout 和部署 planner 都使用相同的动作范围、command smoothing 和 safety envelope。
- 使用 curriculum 覆盖不同赛道位置。
- 定期运行官方评估脚本，而不是只相信 PPO 训练 reward。
- 如果最终卡在弯道，可在不重训低层的前提下先降低 `max_vx`，再逐步提高速度上限。

## 新增/修改的代码

### `track_bonus/planner.py`

新增 `ppo_mlp` planner：

- 读取 `planner_config.json`。
- 加载 `planner_weights.npz`。
- 用 NumPy MLP 做部署推理。
- 输入严格为官方 5D。
- 输出 `[vx, vy, yaw_rate]`。
- 内置 stand phase、safety envelope、command smoothing。

### `train_highlevel_ppo_torch.py`

新增 PyTorch PPO 训练脚本：

- 固定低层 checkpoint。
- 使用 JAX/MJX 跑并行环境。
- PyTorch actor-critic 训练高层。
- 每个 update 导出：
  - `planner_weights.npz`
  - `planner_config.json`
  - `ppo_actor_critic.pt`
  - `training_history.json`
- 可选定期调用 `run_track_bonus.py --no-render` 做官方式评估。

## Colab 运行步骤

本项目实际使用根目录的 `track_bonus_colab_file-2.ipynb` 作为 Colab 入口。该 notebook 已被更新为默认调用 `train_highlevel_ppo_torch.py` 训练高层 PPO planner。注意：notebook 会从 `COURSE_REPO_URL` clone GitHub 仓库到 `/content/go2_track_bonus_repo`，因此运行前需要保证 GitHub 仓库里已经包含 `train_highlevel_ppo_torch.py`、修改后的 `track_bonus/planner.py` 和本文档。如果 Colab 已经 clone 过旧仓库，把配置 cell 里的 `RESET_COURSE_REPO` 临时设为 `True` 后重新运行 setup。

### 1. 准备项目

在 Colab 中进入项目目录。若你已经把本项目上传/clone 到 `/content/Final-Project-Track-2-Bonus-Project`：

```bash
%cd /content/Final-Project-Track-2-Bonus-Project
```

安装依赖：

```bash
!pip install -q -r configs/colab_requirements.txt
```

PyTorch 通常是 Colab 预装的。如果没有：

```bash
!pip install -q torch
```

### 2. 准备低层 checkpoint

本路线不重训低层。把已有低层 checkpoint 放到一个目录，例如：

```text
/content/Final-Project-Track-2-Bonus-Project/best_checkpoint/
```

该目录必须包含：

```text
ppo_network_config.json
_CHECKPOINT_METADATA
...
```

可以先做快速检查：

```bash
!python quick_policy_check.py \
  --checkpoint-dir best_checkpoint \
  --num-steps 100
```

如果用仓库中已有 zip，需要先解压：

```bash
!unzip -q hw1_best_checkpoint.zip -d hw1_best_checkpoint_unzipped
```

然后确认真正 checkpoint 目录在哪一层。

### 3. 训练高层 PPO

短调试：

```bash
!python train_highlevel_ppo_torch.py \
  --checkpoint-dir best_checkpoint \
  --output-dir artifacts/highlevel_ppo_torch_debug \
  --total-updates 3 \
  --num-envs 16 \
  --rollout-steps 128 \
  --eval-interval 0
```

正式训练起点：

```bash
!python train_highlevel_ppo_torch.py \
  --checkpoint-dir best_checkpoint \
  --output-dir artifacts/highlevel_ppo_torch \
  --total-updates 120 \
  --num-envs 64 \
  --rollout-steps 512 \
  --max-episode-seconds 240 \
  --max-vx 1.25 \
  --target-straight-speed 1.10 \
  --target-curve-speed 0.70 \
  --ppo-epochs 4 \
  --minibatch-size 1024 \
  --start-randomization curriculum \
  --eval-interval 0 \
  --eval-seconds 180
```

如果显存不足：

```bash
--num-envs 16 --rollout-steps 128 --minibatch-size 512
```

如果 JAX GPU 出现兼容问题，可以先用 CPU 调试：

```bash
--force-cpu --num-envs 4 --rollout-steps 64
```

### 4. 用官方式脚本评估

训练输出目录里会有 `planner_config.json` 和 `planner_weights.npz`。评估：

```bash
!python run_track_bonus.py \
  --checkpoint-dir best_checkpoint \
  --planner-config artifacts/highlevel_ppo_torch/planner_config.json \
  --output-dir artifacts/highlevel_ppo_torch_eval \
  --duration-seconds 300 \
  --entry-name ppo_mlp
```

只要想快看指标、不渲染视频：

```bash
!python run_track_bonus.py \
  --checkpoint-dir best_checkpoint \
  --planner-config artifacts/highlevel_ppo_torch/planner_config.json \
  --output-dir artifacts/highlevel_ppo_torch_eval \
  --duration-seconds 300 \
  --entry-name ppo_mlp \
  --no-render
```

查看结果：

```bash
!cat artifacts/highlevel_ppo_torch_eval/results.json
```

重点看：

- `metrics.lap_completion`
- `metrics.valid_distance_m`
- `metrics.finish_time`
- `metrics.fall`
- `metrics.boundary_violation`
- `metrics.rms_lateral_error`
- `metrics.mean_progress_speed`

### 5. 提交文件

至少包括：

```text
best_checkpoint/
artifacts/highlevel_ppo_torch/planner_config.json
artifacts/highlevel_ppo_torch/planner_weights.npz
track_bonus/planner.py
train_highlevel_ppo_torch.py
artifacts/highlevel_ppo_torch_eval/results.json
report.pdf
```

报告中建议写清楚：

- 低层 checkpoint 没有重训。
- 高层 actor 输入是官方 5D。
- 训练 reward 使用完整仿真动态信息。
- PPO 的 observation/action/reward/curriculum 设置。
- 与 CEM baseline 的对比。
- 失败案例和调参过程。
