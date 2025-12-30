import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()
for dev in devices:
    print(f"设备名称: {dev.get_info(rs.camera_info.name)}, 序列号: {dev.get_info(rs.camera_info.serial_number)}")