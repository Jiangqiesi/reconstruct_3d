import pyrealsense2 as rs

# 1. 填入你想要使用的摄像头序列号 (可以通过上面的查询获取)
target_serial_number = '123456789012' 
pipeline = rs.pipeline()
config = rs.config()
# 2. 关键：通过序列号启用特定的设备
config.enable_device(target_serial_number)
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
# 这样启动的就是你指定的那个摄像头了
profile = pipeline.start(config)

# 获取深度流的内参
stream = profile.get_stream(rs.stream.depth)
intrinsics = stream.as_video_stream_profile().get_intrinsics()

print(f"宽度: {intrinsics.width}, 高度: {intrinsics.height}")
print(f"焦距 (fx, fy): {intrinsics.fx}, {intrinsics.fy}")
print(f"主点 (ppx, ppy): {intrinsics.ppx}, {intrinsics.ppy}")
print(f"畸变模型: {intrinsics.model}")
print(f"畸变系数: {intrinsics.coeffs}")

pipeline.stop()