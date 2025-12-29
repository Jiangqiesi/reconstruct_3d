# Eye-in-Hand 3D Reconstruction Pipeline

基于机械臂的Eye-in-Hand配置RGB-D相机三维重建系统。

使用 **Realman RM75** 机械臂和 **Intel RealSense** 深度相机，通过 TSDF 融合算法实现高质量物体三维重建。

## 特性

- ✅ **手眼标定** - 支持多种标定算法（Tsai, Park, Horaud等）
- ✅ **轨迹规划** - 球面/螺旋轨迹，LookAt姿态计算
- ✅ **数据采集** - 同步RGB-D图像和机械臂位姿
- ✅ **TSDF融合** - 使用Open3D实现高精度重建
- ✅ **可视化** - 点云、网格、相机轨迹可视化

## 系统要求

- Python 3.8+
- Intel RealSense 相机 (D435/D455等)
- Realman RM75 机械臂
- 棋盘格标定板 (9x6内角点，25mm方格)

## 安装

```bash
# 克隆项目
cd reconstruct_3d

# 安装依赖
pip install -r requirements.txt

# 安装Realman机械臂SDK（从官方获取）
# pip install Robotic_Arm
```

## 快速开始

### 1. 配置参数

编辑 `config/camera_config.yaml`：

```yaml
# 机械臂IP地址
robot:
  ip: "192.168.1.18"
  port: 8080

# 扫描参数
trajectory:
  center: [0.5, 0.0, 0.1]  # 物体中心坐标
  radius: 0.4               # 扫描半径
  num_views: 30             # 视角数量
```

### 2. 手眼标定

```bash
python scripts/run_calibration.py --config config/camera_config.yaml

# 测试模式（无硬件）
python scripts/run_calibration.py --mock
```

操作步骤：
1. 将棋盘格固定在桌面
2. 移动机械臂到不同角度查看棋盘格
3. 按空格键采集样本（约15-20个）
4. 按 'q' 完成采集并计算标定

### 3. 数据采集

```bash
python scripts/run_capture.py \
    --config config/camera_config.yaml \
    --output data/captures/scan_001 \
    --pattern spiral

# 预览轨迹（不执行）
python scripts/run_capture.py \
    --output test \
    --preview

# 测试模式
python scripts/run_capture.py --output test --mock
```

### 4. 三维重建

```bash
python scripts/run_reconstruction.py \
    --data data/captures/scan_001 \
    --output output/model.ply \
    --visualize

# 输出网格
python scripts/run_reconstruction.py \
    --data data/captures/scan_001 \
    --output output/model.obj \
    --mesh \
    --simplify

# 显示相机轨迹
python scripts/run_reconstruction.py \
    --data data/captures/scan_001 \
    --output output/model.ply \
    --visualize \
    --show-trajectory
```

## 项目结构

```
reconstruct_3d/
├── config/
│   └── camera_config.yaml    # 配置文件
├── src/
│   ├── calibration/          # 手眼标定模块
│   │   ├── hand_eye_calibration.py
│   │   └── calibration_utils.py
│   ├── trajectory/           # 轨迹规划模块
│   │   └── trajectory_planner.py
│   ├── capture/              # 数据采集模块
│   │   ├── data_collector.py
│   │   └── realsense_capture.py
│   ├── robot/                # 机械臂接口
│   │   └── realman_robot.py
│   └── reconstruction/       # 重建模块
│       ├── tsdf_fusion.py
│       └── visualization.py
├── scripts/
│   ├── run_calibration.py    # 标定脚本
│   ├── run_capture.py        # 采集脚本
│   └── run_reconstruction.py # 重建脚本
├── data/                     # 数据存储
└── output/                   # 输出结果
```

## API 使用示例

### 手眼标定

```python
from src.calibration import HandEyeCalibrator

# 初始化标定器
calibrator = HandEyeCalibrator(
    camera_matrix=K,
    dist_coeffs=dist,
    pattern_size=(9, 6),
    square_size=0.025
)

# 添加标定样本
calibrator.add_sample(robot_pose, image)

# 执行标定
T_flange_cam = calibrator.calibrate(method='tsai')

# 保存结果
calibrator.save('hand_eye_calibration.npy')
```

### 轨迹规划

```python
from src.trajectory import TrajectoryPlanner

# 创建规划器
planner = TrajectoryPlanner(
    center=[0.5, 0, 0.1],
    radius=0.4,
    T_flange_cam=T_flange_cam
)

# 生成轨迹
camera_poses, flange_poses = planner.generate_spherical_trajectory(
    num_views=30,
    pattern='spiral'
)

# 可视化
planner.visualize_trajectory(camera_poses)
```

### TSDF 重建

```python
from src.reconstruction import TSDFFusion
from src.capture import DataLoader

# 加载数据
loader = DataLoader('data/captures/scan_001')

# 创建融合器
fusion = TSDFFusion(
    intrinsic=loader.get_open3d_intrinsic(),
    voxel_length=0.002,
    sdf_trunc=0.01
)

# 融合所有帧
for color, depth, pose in loader:
    fusion.integrate(color, depth, pose)

# 提取点云
pcd = fusion.extract_point_cloud()
fusion.save_point_cloud('model.ply')
```

## 数据格式

采集的数据以以下格式存储：

```
capture_dir/
├── color/              # RGB图像 (PNG)
│   ├── 000000.png
│   └── ...
├── depth/              # 深度图 (16-bit PNG, 单位mm)
│   ├── 000000.png
│   └── ...
├── poses.txt           # 相机位姿 (4x4矩阵)
├── intrinsics.json     # 相机内参
└── metadata.json       # 采集元数据
```

## 参数调优

### TSDF 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| voxel_length | 0.002m | 体素大小，越小细节越多但内存占用更大 |
| sdf_trunc | 0.01m | SDF截断距离，通常为3-5倍体素大小 |
| depth_trunc | 3.0m | 最大有效深度 |

### 轨迹参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| radius | 0.4m | 扫描半径，根据物体大小调整 |
| num_views | 30 | 视角数量，更多视角覆盖更全面 |
| elevation_range | [20, 70]° | 俯仰角范围 |
| pattern | spiral | 轨迹模式：spiral/layers/uniform |

## 常见问题

### 1. 反光物体效果差

RGB-D相机对反光/透明物体效果差。解决方案：
- 喷涂3D扫描显像剂
- 调整光照条件
- 使用diffuse覆盖物

### 2. 深度图空洞

RealSense有最小工作距离限制（D435约20cm）。确保轨迹半径足够大。

### 3. 重建结果漂移

- 检查手眼标定精度
- 确保机械臂末端稳定
- 增加标定样本数量

## 许可证

MIT License
