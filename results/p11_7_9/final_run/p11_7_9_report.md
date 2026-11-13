# P11.7.9 Fresh Local SLAM → Physical Reverse Parking

状态 **COMPLETE_PARKING_PASS**。真实物理运动：True；最终完整入库：True。

全新 24×24 m、0.05 m 局部地图，初始后轴 (0,0,0)，不读取旧地图或固定轨迹。前后 LiDAR 沿用原射线模拟器，本阶段配置 720 束/270°/12 m/0.015 m，以真实密集射线解决细栅格覆盖缝隙，后向屋顶传感器安装在后轴前方 0.6 m 并朝后，消除两个朝外传感器之间的近处盲区；保留各自安装射线起点；编码器提供位移、IMU 提供相对航向、现有 PointToLine ICP 对新局部 submap 配准，沿用原匹配阈值。实际采集 378 组双传感器扫描，不同 0.2 m 观测位置数 35。

车位：{"goal": [-5.588953667673909, 0.5221322914578199, -0.06841098806568105], "center": [-4.341877566626082, 0.4366852423844333], "slot_polygon": [[-6.861019802713958, -0.7004201818791531], [-2.0019499476727503, -1.0344083558960566], [-1.8225464159455698, 1.5756669722781462], [-6.681616270986778, 1.9096551462950497]], "width_m": 2.616233713936861, "observed_depth_m": 4.2705346684479455, "support_counts": [78, 54], "source": "Fresh occupied long-line pair only; no historical target/scene obstacle coordinates.", "score": 0.016233713936860905}。仅使用新地图长直线边界识别；支持地图两次点击（车身中心、车头方向）并确认，不硬编码旧目标。

探索阶段有实际物理前进；自动泊车只在地图 footprint、定位预算与连续扫掠检查通过后开始。复用 Hybrid A*、ParkingStateMachine、ForwardTracker、ReverseTracker、Stanley/PID 和 PhysicsVehicle。编码器/IMU/局部 SLAM 状态进入控制器，世界位姿仅传感器仿真及离线评估。车只创建一次，探索结束不重置/重建。

安全预算为显式工程概率裕度，不宣称 Gaussian 三西格玛为确定性硬界。执行包络 0.13 m 保留；细栅格精确矩形碰撞、每运动原语半区间位移覆盖和连续插值净空检查覆盖扫掠。旧算法参数及车辆尺寸未改。

最终位置误差 0.07328345507980673 m，航向误差 0.028279816248439258°，实际速度 -9.204425424860342e-06 m/s；保持两秒 True；碰撞 0；最后挡位 -1.0。全部验收项：
```json
{
  "physical_motion": true,
  "forward_motion": true,
  "automatic_complete": true,
  "full_body_inside_original_slot": true,
  "final_position": true,
  "final_heading": true,
  "final_actual_reverse": true,
  "stopped": true,
  "hold_2s": true,
  "no_collision": true,
  "no_direct_pose_changes": true,
  "continuous_path_verified": true,
  "safe_shift_events": true
}
```

失败原因：None。完整实际轨迹、估计轨迹、原始双向扫描、换挡日志和规划证书均保留。历史校验：True。

启动：`conda run --no-capture-output -n parkingplanner python scripts/run_p11_7_9_fresh_local_parking.py --gui --hold`。W/S 调整低速、A/D 转向、C 回正、SPACE 制动；P 停车并规划；地图点击车身中心再点车头方向，ENTER 确认；计划就绪 A 自动执行；R 返回探索；Q/ESC 急停退出。`--demo --fast --no-map-window` 在同一个物理车上进行短程编码器终止的探索并调用相同规划/执行流程，不是旧路径回放。输出目录已存在指标时拒绝覆盖。

## 最终独立物理审计

完整验收状态：**COMPLETE_PARKING_PASS**。在控制器和地图更新结束后，使用记录的真实物理状态独立计算，不反馈在线算法。

| 指标 | 实际结果 | 要求 |
|---|---:|---:|
| 最终位置误差 | 0.073283455 m | ≤0.10 m |
| 最终航向误差 | 0.028279816° | ≤2° |
| 最终实际速度 | -0.000009204425 m/s | 绝对值≤0.01 m/s |
| 原始物理停车位完整车身包含 | True | True |
| 新地图识别车位完整车身包含 | True | True |
| 最后实际运动档位 | -1 | -1 |
| 停车保持≥2秒 | True | True |
| PyBullet 碰撞事件 | 0 | 0 |
| 独立完整矩形障碍物重叠采样 | 0 | 0 |
| 在线地图违规 | 0 | 0 |
| 最大实际局部定位误差 | 0.032231584 m | 计划工程预算0.08 m |
| 实际轨迹到参考折线最大距离 | 0.004877144 m | ≤0.10 m |
| 实际轨迹参考航向最大误差 | 0.242652902° | ≤2° |
| 最终实际后轴到规划末节点距离 | 0.042695604 m | 单独记录 |

最近折线距离是几何跟踪指标，不是时间对齐误差；目标位姿精度始终单独按0.10m/2°判断。

参考路径连续净空下界 0.301210996 m；要求工程总裕度 0.277643480 m，包括0.13m执行裕度。预算明确为工程概率模型，不是确定性误差证书。

退化段最长受限编码器/IMU传播 2.697106 m，小于预设3m；原ICP几何阈值未降低，150帧退化扫描仍记为拒绝，不伪装成功匹配。

## 实际阶段与换挡

前进 4.179167s，倒车 66.595833s，总物理时间 75.350000s。本次自动路径所有运动区间均为倒车，前进发生在实际探索阶段；不宣称执行了不必要的自动前进调整。

实际档位变化：
```json
[
  {
    "time_s": 0.5499999999999993,
    "from_gear": -1,
    "to_gear": 1,
    "actual_speed_before_mps": -1.1360371146902762e-05,
    "encoder_speed_before_mps": -1.3929569147039753e-05
  },
  {
    "time_s": 6.087499999999975,
    "from_gear": 1,
    "to_gear": -1,
    "actual_speed_before_mps": -2.5043328616644984e-05,
    "encoder_speed_before_mps": -2.2533773931962637e-05
  }
]
```
两次实际换挡前速度绝对值均低于0.01m/s。路径第0行保留原Hybrid A*起始节点的+1标签，不代表前进运动；interval_gear数组对应实际规划运动区间，均为-1。

## 保留失败记录与实现修正

初次起步覆盖检查记录保留在父目录：未覆盖的自车近处区域被安全拦截。第一轮有效运动记录保留在full_run：真实前进/换挡/倒车后，在平行墙面退化时急停，FAILED，未覆盖。修正局部估计器先验尺度和退化处理后，以独立final_run完成了一次完整物理泊车。没有重跑P10/P11已通过的旧实验，没有将失败改标PASS。

局部SLAM初始位姿为0；旧物理终点只用于创建物理初始条件。原实时地图、旧P10路径、旧目标坐标均未作为在线输入。独立世界变换只用于结束后的评估和截图。历史结果SHA256校验：True。

## 启动与交互

从D:\ParkingPlanner运行交互模式（独立新结果目录，避免覆盖任何已保存结果）：
```powershell
conda run --no-capture-output -n parkingplanner python scripts/run_p11_7_9_fresh_local_parking.py --gui --hold --output-dir results/p11_7_9/interactive_run
```
请切换英文输入法并激活PyBullet窗口：W/S增减目标速度，A/D转向，C回正，SPACE制动；P停车并规划。地图窗口显示新Occupancy Grid、青色扫描点和估计轨迹，可点击车身中心再点击车头方向，ENTER确认重新规划；计划就绪后A开始自主泊车；R返回探索，Q/ESC制动退出。规划不可行会保留地图并允许继续探索或重新选点。只在窗口活动时响应键盘。

本次验证命令为：
```powershell
conda run --no-capture-output -n parkingplanner python scripts/run_p11_7_9_fresh_local_parking.py --gui --demo --fast --no-map-window --output-dir results/p11_7_9/final_run
```
final_run已保存，程序会拒绝覆盖；复现实验需指定新的输出目录。--demo仅给出短距离、由编码器终止的探索速度指令，然后调用相同的新建图/识别/规划/控制流程，不读取旧固定路径；--fast仅取消墙钟等待，物理步长仍为1/240s。
