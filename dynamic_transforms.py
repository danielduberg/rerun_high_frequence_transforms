import rerun as rr
import csv
import time
import numpy as np

from pathlib import Path

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from point_cloud2 import read_points_numpy

rr.init("example2", spawn=True)

bagpath = Path('/Volumes/Dataset/cache/ntu_viral/eee_02/eee_02.bag')
typestore = get_typestore(Stores.ROS1_NOETIC)

# Camera left

cam_left_focal_length = np.array(
    [4.250258563372763e+02, 4.267976260903337e+02], dtype=np.float32)
cam_left_principal_point = np.array(
    [3.860151866550880e+02, 2.419130336743440e+02], dtype=np.float32)
base_to_cam_left = np.array([0.02183084, -0.01312053,  0.99967558,  0.00552943,
                            0.99975965,  0.00230088, -0.02180248, -0.12431302,
                            -0.00201407,  0.99991127,  0.01316761,  0.01614686,
                            0.00000000,  0.00000000,  0.00000000,  1.00000000], dtype=np.float32).reshape(4, 4)

# rr.log("world/base_link/cam_left", rr.Transform3D(
#     translation=base_to_cam_left[:3, 3], mat3x3=base_to_cam_left[:3, :3]), static=True)
rr.log(
    f"world/base_link/cam_left",
    rr.Pinhole(
        width=752,
        height=480,
        focal_length=cam_left_focal_length,
        principal_point=cam_left_principal_point
    ),
    static=True
)

# Camera right

cam_right_focal_length = np.array(
    [4.313364265799752e+02, 4.327527965378035e+02], dtype=np.float32)
cam_right_principal_point = np.array(
    [3.548956286992647e+02, 2.325508916495161e+02], dtype=np.float32)
base_to_cam_right = np.array([-0.01916508, -0.01496218,  0.99970437,  0.00519443,
                              0.99974371,  0.01176483,  0.01934191,  0.1347802,
                              -0.01205075,  0.99981884,  0.01473287,  0.01465067,
                              0.00000000,  0.00000000,  0.00000000,  1.00000000], dtype=np.float32).reshape(4, 4)

# rr.log("world/base_link/cam_right", rr.Transform3D(
#     translation=base_to_cam_right[:3, 3], mat3x3=base_to_cam_right[:3, :3]), static=True)
rr.log(
    f"world/base_link/cam_right",
    rr.Pinhole(
        width=752,
        height=480,
        focal_length=cam_right_focal_length,
        principal_point=cam_right_principal_point
    ),
    static=True
)

# Lidar horizontal

base_to_lidar_hor = np.array([1.0,  0.0,  0.0, -0.050,
                              0.0,  1.0,  0.0,  0.000,
                              0.0,  0.0,  1.0,  0.055,
                              0.0,  0.0,  0.0,  1.000], dtype=np.float32).reshape(4, 4)


# rr.log(f"/world/base_link/lidar_hor", rr.Transform3D(
#     translation=base_to_lidar_hor[:3, 3], mat3x3=base_to_lidar_hor[:3, :3]), static=True)

# Lidar vertical

base_to_lidar_vert = np.array([-1.0,  0.0,  0.0, -0.550,
                              0.0,  0.0,  1.0,  0.030,
                              0.0,  1.0,  0.0,  0.050,
                              0.0,  0.0,  0.0,  1.000], dtype=np.float32).reshape(4, 4)

# rr.log(f"world/base_link/lidar_vert", rr.Transform3D(
#     translation=base_to_lidar_vert[:3, 3], mat3x3=base_to_lidar_vert[:3, :3]), static=True)

# Read bag file

last_position = [0, 0, 0]
last_rotation = [0, 0, 0, 1]

# Create reader instance and open for reading.
with AnyReader([bagpath]) as reader:
    for connection, timestamp, rawdata in reader.messages():
        rr.set_time("time", timestamp=np.datetime64(timestamp, 'ns'))

        rr.log("world/base_link", rr.Transform3D(
            translation=last_position,
            quaternion=last_rotation
        ))

        rr.log("world/base_link/cam_left", rr.Transform3D(
            translation=base_to_cam_left[:3, 3], mat3x3=base_to_cam_left[:3, :3]))
        rr.log("world/base_link/cam_right", rr.Transform3D(
            translation=base_to_cam_right[:3, 3], mat3x3=base_to_cam_right[:3, :3]))
        rr.log(f"/world/base_link/lidar_hor", rr.Transform3D(
            translation=base_to_lidar_hor[:3, 3], mat3x3=base_to_lidar_hor[:3, :3]))
        rr.log(f"world/base_link/lidar_vert", rr.Transform3D(
            translation=base_to_lidar_vert[:3, 3], mat3x3=base_to_lidar_vert[:3, :3]))

        if connection.topic == '/leica/pose/relative':
            msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
            last_position = [msg.pose.position.x,
                             msg.pose.position.y, msg.pose.position.z]
            # rr.log("world/base_link", rr.Transform3D(
            #     translation=last_position,
            #     quaternion=last_rotation
            # ))
        elif connection.topic == '/imu/imu':
            msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
            last_rotation = [msg.orientation.x, msg.orientation.y,
                             msg.orientation.z, msg.orientation.w]
            # rr.log("world/base_link", rr.Transform3D(
            #     translation=last_position,
            #     quaternion=last_rotation
            # ))
        elif connection.topic == '/left/image_raw':
            msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
            img_data = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, -1)
            rr.log("world/base_link/cam_left",
                   rr.Image(img_data), static=False)

            rr.log("world/base_link/cam_left", rr.Transform3D(
                translation=base_to_cam_left[:3, 3], mat3x3=base_to_cam_left[:3, :3]))
        elif connection.topic == '/right/image_raw':
            msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
            img_data = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, -1)
            rr.log("world/base_link/cam_right",
                   rr.Image(img_data), static=False)

            rr.log("world/base_link/cam_right", rr.Transform3D(
                translation=base_to_cam_right[:3, 3], mat3x3=base_to_cam_right[:3, :3]))
        elif connection.topic == '/os1_cloud_node1/points':
            msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
            points = read_points_numpy(msg, field_names=["x", "y", "z"])
            rr.log(f"world/base_link/lidar_hor",
                   rr.Points3D(positions=points, colors=[255, 0, 0]))

            rr.log(f"/world/base_link/lidar_hor", rr.Transform3D(
                translation=base_to_lidar_hor[:3, 3], mat3x3=base_to_lidar_hor[:3, :3]))
        elif connection.topic == '/os1_cloud_node2/points':
            msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
            points = read_points_numpy(msg, field_names=["x", "y", "z"])
            rr.log(f"world/base_link/lidar_vert",
                   rr.Points3D(positions=points, colors=[0, 0, 255]))

            rr.log(f"world/base_link/lidar_vert", rr.Transform3D(
                translation=base_to_lidar_vert[:3, 3], mat3x3=base_to_lidar_vert[:3, :3]))


# data_file = "/Users/dduberg/Downloads/ground_truth.csv"

# timestamps = []
# positions = []
# rotations = []
# with open(data_file, newline='') as csvfile:
#     reader = csv.reader(csvfile, delimiter=',')
#     next(reader)
#     for row in reader:
#         timestamps.append(int(row[2]))
#         positions.append((float(row[3]), float(row[4]), float(row[5])))
#         rotations.append((float(row[9]), float(row[6]), float(row[7]), float(row[8])))

# rr.log("sun", rr.Ellipsoids3D(half_sizes=[1, 1, 1], colors=[255, 200, 10], fill_mode="solid"))
# rr.log("sun/planet", rr.Ellipsoids3D(half_sizes=[0.4, 0.4, 0.4], colors=[40, 80, 200], fill_mode="solid"))


# for t, p, r in zip(timestamps, positions, rotations):
#     rr.set_time("time", timestamp=np.datetime64(t, "ns"))
#     rr.log("sun/planet", rr.Transform3D(translation=p, quaternion=r))

    # time.sleep(0.0002)

# transforms = []

# path_to_rrd = '/Volumes/Dataset/cache/ntu_viral/eee_02.rrd'

# with rr.server.Server(datasets={'ntu_viral': [path_to_rrd]}) as server:
#     dataset = server.client().get_dataset('ntu_viral')

#     entity = '/os1_cloud_node1/points'
#     component_col = f'{entity}:Points3D:positions'
#     timeline = 'ros2_timestamp'

#     df = dataset.filter_contents([entity]).reader(index=timeline)

#     for stream in df.select(timeline, component_col).repartition(10).execute_stream_partitioned():
#         for batch in stream:
#             pa = batch.to_pyarrow()
