import numpy as np
import struct
import time
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ROSPublisher:
    def __init__(self, node, cfg):
        self.cfg = cfg
        self.node = node
        self.bridge = CvBridge()

        # Create all required publishers
        self.global_path_publisher = node.create_publisher(Path, '/global_path', 10)
        self.local_path_publisher = node.create_publisher(Path, '/local_path', 10)
        self.action_path_publisher = node.create_publisher(Path, '/action_path', 10)

        self.image_publisher = node.create_publisher(Image, '/annotated_image', 10)
        self.fs_image_publisher = node.create_publisher(Image, '/fastsam_image', 10)
        self.fs_image_after_publisher = node.create_publisher(Image, '/fastsam_image_after', 10)

        self.pose_publisher = node.create_publisher(Odometry, '/odom', 10)

        self.local_rgb_publisher = node.create_publisher(PointCloud2, '/local_map/rgb', 10)
        self.local_sem_publisher = node.create_publisher(PointCloud2, '/local_map/semantic', 10)

        self.global_rgb_publisher = node.create_publisher(PointCloud2, '/global_map/rgb', 10)
        self.global_sem_publisher = node.create_publisher(PointCloud2, '/global_map/semantic', 10)

    def publish_all(self, dualmap):
        """
        Publish all messages: paths, images, poses, and point clouds.
        """
        # 1. Publish paths
        self._publish_path(dualmap.curr_global_path, 'global')
        self._publish_path(dualmap.curr_local_path, 'local')
        self._publish_path(dualmap.action_path, 'action')

        if self.cfg.use_rviz:

            # 2. Publish images
            self._publish_image(dualmap.detector.annotated_image, 'annotated')
            self._publish_image(dualmap.detector.annotated_image_fs, 'fastsam')
            self._publish_image(dualmap.detector.annotated_image_fs_after, 'fastsam_after')

            # 3. Publish pose
            self._publish_pose(dualmap.curr_pose)

            # 4. Publish local map (RGB point cloud + semantic point cloud)
            if len(dualmap.local_map_manager.local_map):
                start_time = time.time()
                self._publish_local_map(dualmap.local_map_manager, dualmap.visualizer, publish_rgb=False)
                print(f"Publishing local map took {time.time() - start_time:.2f} seconds.")

            if len(dualmap.global_map_manager.global_map):
                self._publish_global_map(dualmap.global_map_manager, dualmap.visualizer, publish_rgb=False)

    def _publish_path(self, path, path_type):
        if path is None:
            return

        path_msg = Path()
        path_msg.header.stamp = self.node.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'

        for pos in path:
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            pose_stamped.pose.position.x = pos[0]
            pose_stamped.pose.position.y = pos[1]
            pose_stamped.pose.position.z = pos[2]
            pose_stamped.pose.orientation.w = 1.0
            path_msg.poses.append(pose_stamped)

        publisher = {
            'global': self.global_path_publisher,
            'local': self.local_path_publisher,
            'action': self.action_path_publisher
        }.get(path_type, None)

        if publisher:
            publisher.publish(path_msg)
            # self.node.get_logger().info(f"Published {path_type} path.")

    def _publish_image(self, image, image_type):
        if image is None:
            return

        ros_image = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        publisher = {
            'annotated': self.image_publisher,
            'fastsam': self.fs_image_publisher,
            'fastsam_after': self.fs_image_after_publisher
        }.get(image_type, None)

        if publisher:
            publisher.publish(ros_image)

    def _publish_pose(self, pose_matrix):
        if pose_matrix is None:
            return

        # Extract translation and rotation
        translation = pose_matrix[:3, 3]
        quaternion = self.rotation_matrix_to_quaternion(pose_matrix[:3, :3])

        # Create Odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = self.node.get_clock().now().to_msg()  # Timestamp
        odom_msg.header.frame_id = "map"  # Parent frame (global coordinate system)
        odom_msg.child_frame_id = ""  # Child frame (robot coordinate system)

        # Set position and orientation
        odom_msg.pose.pose.position.x = translation[0]
        odom_msg.pose.pose.position.y = translation[1]
        odom_msg.pose.pose.position.z = translation[2]
        odom_msg.pose.pose.orientation.x = float(quaternion[0])
        odom_msg.pose.pose.orientation.y = float(quaternion[1])
        odom_msg.pose.pose.orientation.z = float(quaternion[2])
        odom_msg.pose.pose.orientation.w = float(quaternion[3])

        # Set velocity information (default to 0)
        odom_msg.twist.twist.linear.x = 0.0
        odom_msg.twist.twist.linear.y = 0.0
        odom_msg.twist.twist.linear.z = 0.0
        odom_msg.twist.twist.angular.x = 0.0
        odom_msg.twist.twist.angular.y = 0.0
        odom_msg.twist.twist.angular.z = 0.0

        # Publish Odometry message
        self.pose_publisher.publish(odom_msg)

    def _publish_local_map(self, local_map_manager, visualizer, publish_rgb=True):
        all_positions = []
        all_rgb_colors = []
        all_semantic_colors = []

        for local_obj in local_map_manager.local_map:
            obj_name = visualizer.obj_classes.get_classes_arr()[local_obj.class_id]
            positions = np.asarray(local_obj.pcd.points)
            colors = (np.asarray(local_obj.pcd.colors) * 255).astype(np.uint8)
            curr_obj_color = np.array(visualizer.obj_classes.get_class_color(obj_name)) * 255
            curr_obj_color = curr_obj_color.astype(np.uint8)
            semantic_colors = np.tile(curr_obj_color, (positions.shape[0], 1))

            all_positions.append(positions)
            all_rgb_colors.append(colors)
            all_semantic_colors.append(semantic_colors)

        if not all_positions:
            return

        all_positions = np.vstack(all_positions)
        all_rgb_colors = np.vstack(all_rgb_colors)
        all_semantic_colors = np.vstack(all_semantic_colors)

        if publish_rgb:
            self.publish_pointcloud(all_positions, all_rgb_colors, self.local_rgb_publisher, "map")

        self.publish_pointcloud(all_positions, all_semantic_colors, self.local_sem_publisher, "map")

    def _publish_global_map(self, global_map_manager, visualizer, publish_rgb=True):
        all_positions = []
        all_rgb_colors = []
        all_semantic_colors = []

        for global_obj in global_map_manager.global_map:
            obj_name = visualizer.obj_classes.get_classes_arr()[global_obj.class_id]
            positions = np.asarray(global_obj.pcd_2d.points)
            colors = (np.asarray(global_obj.pcd_2d.colors) * 255).astype(np.uint8)
            curr_obj_color = np.array(visualizer.obj_classes.get_class_color(obj_name)) * 255
            curr_obj_color = curr_obj_color.astype(np.uint8)
            semantic_colors = np.tile(curr_obj_color, (positions.shape[0], 1))

            all_positions.append(positions)
            all_rgb_colors.append(colors)
            all_semantic_colors.append(semantic_colors)

        if not all_positions:
            return

        all_positions = np.vstack(all_positions)
        all_rgb_colors = np.vstack(all_rgb_colors)
        all_semantic_colors = np.vstack(all_semantic_colors)

        if publish_rgb:
            self.publish_pointcloud(all_positions, all_rgb_colors, self.global_rgb_publisher, "map")

        self.publish_pointcloud(all_positions, all_semantic_colors, self.global_sem_publisher, "map")

    def publish_pointcloud(self, points, colors, publisher, frame_id):
        """
        Efficient point cloud publishing: batch pack RGB data with NumPy
        """
        num_points = points.shape[0]

        # Efficient RGB packing: avoid Python loops
        r = colors[:, 0].astype(np.uint32)
        g = colors[:, 1].astype(np.uint32)
        b = colors[:, 2].astype(np.uint32)
        rgb_packed = (r << 16) | (g << 8) | b  # RGB → UINT32

        # Efficient data concatenation: build PointCloud data at once
        cloud_data = np.zeros((num_points, 4), dtype=np.float32)
        cloud_data[:, :3] = points
        cloud_data[:, 3] = rgb_packed.view(np.float32)  # direct conversion

        # PointCloud2 field definition
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)
        ]

        # Message header
        header = Header()
        header.stamp = self.node.get_clock().now().to_msg()
        header.frame_id = frame_id

        # Build PointCloud2 message
        pointcloud_msg = PointCloud2()
        pointcloud_msg.header = header
        pointcloud_msg.height = 1
        pointcloud_msg.width = num_points
        pointcloud_msg.fields = fields
        pointcloud_msg.is_bigendian = False
        pointcloud_msg.point_step = 16
        pointcloud_msg.row_step = 16 * num_points
        pointcloud_msg.is_dense = True
        pointcloud_msg.data = cloud_data.tobytes()

        # Publish point cloud
        publisher.publish(pointcloud_msg)

    def rotation_matrix_to_quaternion(self, R):
        """
        Convert a rotation matrix to a quaternion.

        Parameters:
        - R: A 3x3 rotation matrix.

        Returns:
        - A quaternion in the format [x, y, z, w].
        """
        # Make sure the matrix is a numpy array
        R = np.asarray(R)
        # Allocate space for the quaternion
        q = np.empty((4,), dtype=np.float32)
        # Compute the quaternion components
        q[3] = np.sqrt(np.maximum(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
        q[0] = np.sqrt(np.maximum(0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
        q[1] = np.sqrt(np.maximum(0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
        q[2] = np.sqrt(np.maximum(0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
        q[0] *= np.sign(q[0] * (R[2, 1] - R[1, 2]))
        q[1] *= np.sign(q[1] * (R[0, 2] - R[2, 0]))
        q[2] *= np.sign(q[2] * (R[1, 0] - R[0, 1]))

        return q
