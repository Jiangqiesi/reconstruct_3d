import pyrealsense2 as rs

# 先列出所有可用设备
ctx = rs.context()
devices = ctx.query_devices()
print(f"找到 {len(devices)} 个设备:")
for dev in devices:
    print(f"  序列号: {dev.get_info(rs.camera_info.serial_number)}")
    print(f"  名称: {dev.get_info(rs.camera_info.name)}")

# 1. 填入你想要使用的摄像头序列号
target_serial_number = '427622270277' 
pipeline = rs.pipeline()
config = rs.config()

# 2. 关键:通过序列号启用特定的设备
config.enable_device(target_serial_number)

# 3. 尝试使用更通用的配置或先不指定分辨率
# config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.depth)  # 使用默认配置

try:
    profile = pipeline.start(config)
    
    # 获取深度流的内参
    stream = profile.get_stream(rs.stream.depth)
    intrinsics = stream.as_video_stream_profile().get_intrinsics()
    
    print(f"\n宽度: {intrinsics.width}, 高度: {intrinsics.height}")
    print(f"焦距 (fx, fy): {intrinsics.fx}, {intrinsics.fy}")
    print(f"主点 (ppx, ppy): {intrinsics.ppx}, {intrinsics.ppy}")
    print(f"畸变模型: {intrinsics.model}")
    print(f"畸变系数: {intrinsics.coeffs}")
    
    pipeline.stop()
    
except RuntimeError as e:
    print(f"启动失败: {e}")
    print("\n可能的原因:")
    print("1. 设备序列号不正确")
    print("2. 设备不支持 848x480 @ 30fps 的深度流")
    print("3. 设备已被其他程序占用")
    print("4. USB 连接不稳定或带宽不足")