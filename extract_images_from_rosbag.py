
import argparse
import cv2
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

import os


def compressed_image_msg_to_cv2(image_msg):
    """Convert a sensor_msgs/Image message to an OpenCV image (NumPy array)."""
    np_arr = np.frombuffer(image_msg.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # Use IMREAD_COLOR to handle color images
    if 'rgb8' in image_msg.format:
        return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    elif 'bgr8' in image_msg.format:
        return image_np
    elif 'mono8' in image_msg.format or 'mono16' in image_msg.format:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    return image_np


def image_msg_to_cv2(image_msg):
    # Convert the image data to a NumPy array
    dtype = np.uint8 if image_msg.encoding == 'mono8' else np.uint16 if image_msg.encoding == 'mono16' else np.uint8
    image_np = np.frombuffer(image_msg.data, dtype=dtype).reshape(image_msg.height, image_msg.width, -1)

    # Handle different encodings
    if image_msg.encoding == 'rgb8':
        return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    elif image_msg.encoding == 'bgr8':
        return image_np
    elif image_msg.encoding == 'mono8' or image_msg.encoding == 'mono16':
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    return image_np


def parse_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Extract images from ROS bag.")

    # Add arguments
    parser.add_argument('bag_path', type=str,
                        help='Path to the ROS bag file.')
    parser.add_argument('--image_topic', type=str, default='/thermal/image_raw',
                        help='ROS topic to extract images from.')
    parser.add_argument('--outputdir', type=str, default='/home/pi/Desktop/temp/thermal',
                        help='Directory to save the extracted images.')

    # Parse the arguments
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Display the parsed arguments (for debugging purposes)
    print(f"Bag Path: {args.bag_path}")
    print(f"Image Topic: {args.image_topic}")
    print(f"Output Directory: {args.outputdir}")

    os.makedirs(args.outputdir, exist_ok=True)

    typestore = get_typestore(Stores.ROS1_NOETIC)
    # Open the ROS1 bag
    with Reader(args.bag_path) as reader:
        # Topic and msgtype information is available on .connections list.
        print('rosbag info:')
        for connection in reader.connections:
            print(connection.topic, connection.msgtype)

        # Iterate over messages in the bag
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == args.image_topic:
                # Deserialize the raw data into a sensor_msgs/Image message
                img_msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                if hasattr(img_msg, 'format') and 'compressed' in img_msg.format:
                    cv_image = compressed_image_msg_to_cv2(img_msg)
                else:
                    cv_image = image_msg_to_cv2(img_msg)

                to_uint8 = True
                if to_uint8:
                    cv_image = cv_image.astype(np.float32)

                    # Normalize to range [0, 255]
                    cv_image_min = np.min(cv_image)
                    cv_image_max = np.max(cv_image)

                    # Avoid division by zero if the image has the same value everywhere
                    if cv_image_max > cv_image_min:
                        cv_image = (cv_image - cv_image_min) / (cv_image_max - cv_image_min) * 255
                    else:
                        cv_image = np.zeros_like(cv_image)  # All pixels are the same, set to zero
                    cv_image = cv_image.astype(np.uint8)
                # Generate a filename based on the timestamp
                timestr = '{}.{:09d}.png'.format(img_msg.header.stamp.sec,
                                                 img_msg.header.stamp.nanosec)
                filename = os.path.join(args.outputdir, timestr)
                cv2.imwrite(filename, cv_image)
                print(f"Saved {filename}")


if __name__ == '__main__':
    main()
