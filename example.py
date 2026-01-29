import pathlib
import time
from tracemalloc import start
from typing_extensions import Final
import rerun as rr
import numpy as np
import argparse

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from point_cloud2 import read_points_numpy

EXAMPLE_DIR: Final = pathlib.Path(__file__).parent
DATASET_FILE: Final = EXAMPLE_DIR / "eee_02_sample.bag"


def log_dataset(bagpath: pathlib.Path, static_transforms: bool) -> None:

    # Pinhole camera models

    rr.log(
        f"world/base_link/cam_left",
        rr.Pinhole(
            width=752,
            height=480,
            focal_length=[425.0258563372763, 426.7976260903337],
            principal_point=[386.015186655088, 241.913033674344],
        ),
        static=True
    )

    rr.log(
        f"world/base_link/cam_right",
        rr.Pinhole(
            width=752,
            height=480,
            focal_length=[431.3364265799752, 432.7527965378035],
            principal_point=[354.8956286992647, 232.5508916495161]
        ),
        static=True
    )

    # Camera left

    b2cl_pos = [0.00552943, -0.12431302, 0.01614686]
    b2cl_rot = np.array([[0.02183084, -0.01312053,  0.99967558],
                         [0.99975965,  0.00230088, -0.02180248],
                         [-0.00201407,  0.99991127,  0.01316761]], dtype=np.float32)

    # Camera right

    b2cr_pos = [0.00519443, 0.1347802, 0.01465067]
    b2cr_rot = np.array([[-0.01916508, -0.01496218,  0.99970437],
                         [0.99974371,  0.01176483,  0.01934191],
                         [-0.01205075,  0.99981884,  0.01473287]], dtype=np.float32)

    # Lidar horizontal

    b2lh_pos = [-0.050, 0.000, 0.055]
    b2lh_rot = np.array([[1.0,  0.0,  0.0],
                         [0.0,  1.0,  0.0],
                         [0.0,  0.0,  1.0]], dtype=np.float32)

    # Lidar vertical

    b2lv_pos = [-0.550, 0.030, 0.050]
    b2lv_rot = np.array([[-1.0,  0.0,  0.0],
                         [0.0,  0.0,  1.0],
                         [0.0,  1.0,  0.0]], dtype=np.float32)

    # Read bag file

    typestore = get_typestore(Stores.ROS1_NOETIC)

    last_position = [0, 0, 0]
    last_rotation = [0, 0, 0, 1]

    if static_transforms:
        rr.log("world/base_link/cam_left",
               rr.Transform3D(translation=b2cl_pos, mat3x3=b2cl_rot), static=True)
        rr.log("world/base_link/cam_right",
               rr.Transform3D(translation=b2cr_pos, mat3x3=b2cr_rot), static=True)
        rr.log(f"/world/base_link/lidar_hor",
               rr.Transform3D(translation=b2lh_pos, mat3x3=b2lh_rot), static=True)
        rr.log(f"world/base_link/lidar_vert",
               rr.Transform3D(translation=b2lv_pos, mat3x3=b2lv_rot), static=True)

    start = time.time()
    with AnyReader([bagpath]) as reader:
        for connection, timestamp, rawdata in reader.messages():
            rr.set_time("time", timestamp=np.datetime64(timestamp, 'ns'))

            rr.log("world/base_link", rr.Transform3D(
                translation=last_position,
                quaternion=last_rotation
            ))

            if not static_transforms:
                rr.log("world/base_link/cam_left",
                       rr.Transform3D(translation=b2cl_pos, mat3x3=b2cl_rot))
                rr.log("world/base_link/cam_right",
                       rr.Transform3D(translation=b2cr_pos, mat3x3=b2cr_rot))
                rr.log(f"/world/base_link/lidar_hor",
                       rr.Transform3D(translation=b2lh_pos, mat3x3=b2lh_rot))
                rr.log(f"world/base_link/lidar_vert",
                       rr.Transform3D(translation=b2lv_pos, mat3x3=b2lv_rot))

            if connection.topic == '/leica/pose/relative':
                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                last_position = [msg.pose.position.x,
                                 msg.pose.position.y, msg.pose.position.z]
            elif connection.topic == '/imu/imu':
                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                last_rotation = [msg.orientation.x, msg.orientation.y,
                                 msg.orientation.z, msg.orientation.w]
            elif connection.topic == '/os1_cloud_node1/points':
                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                points = read_points_numpy(msg, field_names=["x", "y", "z"])
                rr.log(f"world/base_link/lidar_hor",
                       rr.Points3D(positions=points, colors=[255, 0, 0]))

    end = time.time()
    print(f"It took: {end - start} seconds")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rosbag",
        type=pathlib.Path,
        default=DATASET_FILE,
        help="Dataset directory",
    )
    parser.add_argument(
        "--static_transforms",
        action="store_true",
        help="Log static transforms only",
    )
    rr.script_add_args(parser)
    args = parser.parse_args()

    rr.script_setup(args, "high_frequence_transforms_example")

    log_dataset(args.rosbag, args.static_transforms)

    rr.script_teardown(args)


if __name__ == "__main__":
    main()
