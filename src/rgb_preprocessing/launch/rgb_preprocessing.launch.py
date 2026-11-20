from launch import LaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = {
        'rectify': 'true',
        'raw_image': '/oakd/rgb/preview/image_raw',
        'camera_info': '/oakd/rgb/preview/camera_info',
        'rect_image': '/oakd/rgb/preview/image_rect',
        'clean_image': '/oakd/rgb/preview/image_rect_clean',
        'valid_mask': '/rgb_rectification/valid_mask',
        'edge_guard_px': '8',
        'margin_px': '3',
        'inpaint_radius': '3.0',
    }
    return LaunchDescription([
        *[DeclareLaunchArgument(k, default_value=v) for k, v in args.items()],
        Node(package='image_proc', executable='rectify_node', name='rgb_rectifier',
             condition=IfCondition(LaunchConfiguration('rectify')),
             remappings=[('image', LaunchConfiguration('raw_image')),
                         ('camera_info', LaunchConfiguration('camera_info')),
                         ('image_rect', LaunchConfiguration('rect_image'))]),
        Node(package='rgb_preprocessing', executable='rgb_border_cleaner',
             name='rgb_border_cleaner', output='screen', parameters=[{
                'input_topic': LaunchConfiguration('rect_image'),
                'camera_info_topic': LaunchConfiguration('camera_info'),
                'output_topic': LaunchConfiguration('clean_image'),
                'valid_mask_topic': LaunchConfiguration('valid_mask'),
                'edge_guard_px': ParameterValue(LaunchConfiguration('edge_guard_px'), value_type=int),
                'margin_px': ParameterValue(LaunchConfiguration('margin_px'), value_type=int),
                'inpaint_radius': ParameterValue(LaunchConfiguration('inpaint_radius'), value_type=float),
             }]),
    ])
