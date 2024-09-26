
import cv2
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

import os

def image_msg_to_cv2(image_msg):
    """Convert a sensor_msgs/Image message to an OpenCV image (NumPy array)."""
    # Convert the image data to a NumPy array
    dtype = np.uint8 if image_msg.encoding == 'mono8' else np.uint16 if image_msg.encoding == 'mono16' else np.uint8
    image_np = np.frombuffer(image_msg.data, dtype=dtype).reshape(image_msg.height, image_msg.width, -1)

    # Handle different encodings
    if image_msg.encoding == 'rgb8':
        return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    elif image_msg.encoding == 'bgr8':
        return image_np
    elif image_msg.encoding == 'mono8' or image_msg.encoding == 'mono16':
        return cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    return image_np


# Path to your bag file
bag_path = '/media/pi/MyBookDuo/jhuai/data/vividpp/global_robust.bag'

# Topic from which to extract images
image_topic = '/thermal/image_raw'
image_topic = '/rgb/image'
outputdir = '/home/pi/Desktop/temp/thermal8'

typestore = get_typestore(Stores.ROS1_NOETIC)
# Open the ROS1 bag
with Reader(bag_path) as reader:
    # Iterate over messages in the bag
    for connection, timestamp, rawdata in reader.messages():
        if connection.topic == image_topic:
            # Deserialize the raw data into a sensor_msgs/Image message
            img_msg = typestore.deserialize_ros1(rawdata, connection.msgtype)

            # Convert the ROS image message to a cv2-compatible image
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
            filename = os.path.join(outputdir, timestr)

            # Save the image as a JPG file
            cv2.imwrite(filename, cv_image)

            print(f"Saved {filename}")
