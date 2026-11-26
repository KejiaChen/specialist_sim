# FR3 单臂轨迹回放：SciPy、Omega 限速 IK 与 Pink

## 双桌布局

`src/specialist_sim/configs/fr3_omega.toml` 默认启用双桌。底座桌面为
80 × 120 cm，离地 75 cm。FR3 底座距朝外的短边约 5 mm，`home` 姿态
伸向 60 × 120 cm 的操作桌，操作桌面离地 94 cm。双桌随 FR3 的标定基座
定位，地面位于基座安装面下方 75 cm。

在 `[layout.tables]` 中设置 `work_y` 可横移操作桌，设置
`work_yaw_deg` 可绕竖直轴逆时针旋转。默认相对角度为 90°。操作桌的
纵向位置会随角度调整，使两桌的纵向间距保持 4 cm。也可设置 `work_x`
直接指定操作桌中心的纵向位置。坐标以底座桌中心为原点，+X 指向操作桌，
+Y 指向左侧，距离单位是米。

可在 PowerShell 中直接预览 `home` 姿态。当前配置从
`local/models/specialist_omega_description/acuated_new.xml` 读取由
`acuated_new.urdf` 转换的夹爪。模型和它引用的网格应一起保留。
使用其他 MJCF 时可修改 `layout.gripper_mjcf` 或传入 `--gripper-mjcf`。
源 CAD 导出保留了关节运动学，夹爪惯量由网格推算，行程沿用回放约定。

```powershell
uv run --locked --all-extras sim-replay-arm --config src/specialist_sim/configs/fr3_omega.toml --preview-home local/fr3-home.jpg --no-preview-window
```

## 换一条轨迹：先设置回放参数

[FR3_replay.sh](FR3_replay.sh) 开头定义了默认参数。每次选一个后端运行，换数据时只传需要覆盖的选项。例如，在仓库根目录用 Pink 回放另一条轨迹（将示例路径和 ID 换成实际值）：

```bash
./FR3_replay.sh --pink \
  --dataset "/path/to/trial" \
  --session "YOUR_SESSION_ID"
```

| 命令行选项 | 脚本默认值 | 说明 |
|---|---|---|
| `--dataset` | `../specialist_datasets/pick_cup_02/processed/single/trial_261117_002` | 包含状态层的 trial 目录 |
| `--session` | `20261117_182618_IAQEB260436VUM` | 该 trial 中的 session ID，省略 `.rrd` 后缀 |
| `--layer` | `state_s8_r100` | 指定重建状态层 |
| `--hand` | `right` | 回放 `left` 或 `right` |
| `--config` | `src/specialist_sim/configs/fr3_omega.toml` | FR3 基座和夹爪安装配置 |
| `--output-root` | `local/replay` | 保存每次回放的根目录 |

用 `./FR3_replay.sh --help` 查看用法。脚本按仓库根目录解析相对路径，自动传入选定后端的源码目录和 Python 解释器；换机器时可在脚本开头修改这四个环境路径。输出位于 `<output-root>/<session>/<hand>/<后端>.<随机后缀>/`，启动时会打印具体目录。默认生成视频，加 `--no-video` 可只生成 CSV 和 JSON。

`sim-replay-arm` 从录制的 `state_*` 层逐帧读取时间戳、夹爪 base tag 的目标位姿和夹爪开合。三个后端都用七个臂关节追踪同一个 tag；夹爪开合按录制值单独回放。SciPy 和 Omega 每帧直接求关节角，Pink 从第二帧起求关节增量。三者都使用加权目标，姿态偏好与末端跟踪之间允许折中。

下文用 $q$ 表示七个关节角（rad），$q_h$ 表示 FR3 模型的 `home`，$q_-$ 表示上一帧**实际输出**的关节角，$\Delta t$ 取相邻录制时间戳之差。$e_p(q)$ 是 tag 的位置误差（m），$\theta(q)$ 是实际朝向到目标朝向的最短转角（rad）。公式使用当前默认参数，省略不影响最优解的整体常数因子。

运行结束后，`frames.csv` 是逐帧结果，`summary.json` 记录实际布局、参数和统计，`replay.mp4` 是视频。比较时看 `passed` 和 `over_speed_frames` 两项：通过帧要求求解收敛、位置误差 ≤ 5 mm、旋转误差 ≤ 3°、未保持旧解且无检测到的碰撞；超速单独统计。

## 1. SciPy：有界最小二乘直接 IK

**计算方式。** 从参考关节角 $q_s$ 出发，SciPy 的 `least_squares` 尝试候选 $q$，MuJoCo 做正运动学并计算候选姿态下的 tag 误差。首帧 $q_s=q_h$。之后只有求解收敛且位姿达标的有效解才更新 $q_s$。因此，超差帧可以输出一个有限解，而下一帧仍从更早的有效解开始。每帧的默认函数评估预算为 35。

**目标函数。** 在模型给出的 $q_{\min}\le q\le q_{\max}$ 内最小化

$$
C_{\rm scipy}(q)=
\frac{\|e_p(q)\|^2}{(0.005\,\mathrm{m})^2}
+\frac{\theta(q)^2}{(3\pi/180\,\mathrm{rad})^2}
+0.02\|q-q_h\|^2
+0.1\|q-q_s\|^2 .
$$

前两项按验收容差归一化：位置恰好偏 5 mm 或朝向恰好偏 3°，相应项都贡献 1。第三项偏好靠近 home 的手臂姿态，第四项鼓励解延续上一次有效构型。第四项只惩罚角度变化，首末两个关节位置各自合法也可能产生很大的帧间速度。SciPy 会报告这种跳变，求解时的硬约束只有角度范围。[SciPy 求解与回放实现](src/specialist_sim/arm.py)

```bash
./FR3_replay.sh --scipy
```

## 2. Omega-IK limited：以 FR3 home 为参考的限速直接 IK

**计算方式与 FR3 改动。** Omega-Infra 的 H1-2 用 Pinocchio 做符号正运动学、CasADi 建模、IPOPT 求关节角。我们的 `dev-fr3` 分支保留了这个结构和原来的四帧滤波器，做了三处影响回放结果的适配：

1. **目标改为单臂 tag。** 从组装好的 MuJoCo 模型导出 FR3 七轴运动学链，以及法兰到 tag 的固定变换。将重建目标转换到机械臂基座系，保持米制尺度。启动时用 MuJoCo 独立核对 Pinocchio 的法兰和 tag 位姿。
2. **关节姿态参考从零位改为模型 home。** 原 H1-2 正则项为 $0.02\|q\|^2$，FR3 改为 $0.02\|q-q_h\|^2$。当前模型的 home 按 J1–J7 约为 $[0,0,0,-90,0,90,-45]$ 度。求解器从模型读取该向量；它在整个回放中固定，而连续性项的参考随帧更新。
3. **将速度限额施加到最终输出。** 首帧以 home 热启动，只有角度边界。从第二帧起，以 $q_-$ 为中心，把角度边界与 $[q_- - v_{\max}\Delta t,\ q_- + v_{\max}\Delta t]$ 逐轴取交集。IPOPT 在交集内求解，再用原四帧滤波（权重从新到旧为 0.4、0.3、0.2、0.1），最后将滤波结果也投影到同一区间。求解失败时保持上一实际输出。

**目标函数。** 首帧在下式中取 $q_-=q_h$，随后取上一帧最终输出：

$$
C_{\rm omega}(q)=
50\|e_p(q)\|^2+\theta(q)^2
+0.02\|q-q_h\|^2+0.1\|q-q_-\|^2 .
$$

四项依次衡量位置、朝向、偏离 home 的程度和一步的关节变化。与 SciPy 相比，这里的位置和朝向直接使用米和弧度：5 mm 的位置误差只贡献 $50(0.005)^2=0.00125$。因此，即使两个后端的 home 系数都写作 0.02，其相对于末端跟踪的影响也不同。FR3 的旋转项用等价的最短转角平方计算，以处理共轴目标附近的数值求导。限速属于硬约束，独立于连续性权重。[FR3 回放适配](src/specialist_sim/omega_ik.py) · [本机 dev-fr3 求解器](/home/kejia/Documents/Omega-Infra-dev-fr3/hal/arm/deploy/robot_control/fr3_arm_ik.py)

```bash
./FR3_replay.sh --omega-limited
```

## 3. Pink：每个采样区间解一次微分 IK

**计算方式。** 首帧调用第 1 节的 SciPy IK，从 home 定位到录制起点。后续每帧从 $q_-$ 出发，将当前 tag 到目标的位姿误差线性化，使用 DAQP 求一个关节增量 $\Delta q$。Pink 返回速度 $\Delta q/\Delta t$，回放按同一个 $\Delta t$ 积分得到新姿态。QP 同时施加模型关节位置约束和 $|\Delta q_i|\le v_{\max,i}\Delta t$。Pink 与 Omega limited 使用相同的逐轴速度上限；Pink 每步积分后直接输出。

**目标函数。** 令 $e=\log(T_{\rm tag}^{-1}T^*)$ 是目标相对当前 tag 的六维局部位姿误差，$J_e$ 是**该误差**对关节角的雅可比。默认任务增益为 1，因此一步后的预测误差为 $e+J_e\Delta q$。把三个任务及数值阻尼展开，固定基座 FR3 的每步目标为

$$
\begin{aligned}
C_{\rm pink}(\Delta q)={}&
\|W(e+J_e\Delta q)\|^2
+ (10^{-3})^2\|q_-+\Delta q-q_h\|^2 \\
&+ (10^{-3})^2\|\Delta q\|^2
+ (\mu+10^{-12})\|\Delta q\|^2 .
\end{aligned}
$$

- **FrameTask** 同时追踪位置和朝向。$W$ 的前三个分量取 `position_cost`，后三个取 `orientation_cost`，默认都为 1。它优化本步线性化后的六维残差。
- **PostureTask** 让更新后的关节姿态略微偏向 FR3 home。Pink 先将残差乘以 `posture_cost=0.001` 再平方，所以这里的平方项系数是 $10^{-6}$。
- **DampingTask** 以 `damping_cost=0.001` 惩罚本步关节增量。除此之外，FrameTask 还加入随误差变化的 LM 阻尼 $\mu=\texttt{lm\_damping}\,\|We\|^2$，默认 `lm_damping=1`；$10^{-12}$ 是 QP 的全局数值阻尼。三个任务加权求和，末端跟踪和姿态偏好之间没有显式的零空间优先级。[Pink 回放任务配置](src/specialist_sim/scripts/pink_worker.py) · [本机 Pink 任务代价实现](/home/kejia/Documents/pink/pink/tasks/task.py)

```bash
./FR3_replay.sh --pink
```

三个后端的公共位姿容差和按关节名定义的速度限额可在 TOML 的 `[ik]` 中配置；Omega、Pink 的求解参数分别在 `[omega]`、`[pink]` 中覆盖。对比实验时，先核对各次 `summary.json` 中的 `layout.world_from_base`、`state_path` 和实际参数，再比较 `passed`、位置与旋转误差的 P95，以及 `over_speed_frames`。
