#!/usr/bin/env python
# flake8: noqa
# pylint: disable=W0106

import collections
import os
import rospkg
import shutil
import sys
import csv
import utm
import logging
import json
import rosbag

from optparse import OptionParser
from pyquaternion import Quaternion

logger = logging.getLogger(__name__)



def lat_lon_list_to_x_y(lat, lon):
  """Convert lat/long coordinates to utm coordinates."""
  x, y = [], []
  last_x1, last_y1 = 0, 0
  for val in zip(lat, lon):
    try:
      x1, y1, _, _ = utm.from_latlon(val[0], val[1])
      last_x1, last_y1 = x1, y1
    except:
      logger.error("Exception {} encountered for lat, lon: {}".format(
          sys.exc_info()[0], val))
      x1, y1 = last_x1, last_y1
    x.append(x1)
    y.append(y1)

  return x, y

def read_topics_from_ros_bag(ros_bag_file,
                             topics=None,
                             start_time=None,
                             end_time=None):
  """Reads different kinds of topics from a ros bag file. The topics will be one
  or more of followings:

  /pose
  /localization/gps_pose
  /localization/ins_pose
  /gps_imu_data

  Returns a dict with key as pose name, value as pose array
  pose:[pose1, pose2...]
  lidar_pose:[pose3, pose4...]
  lidar_meas:[meas1, meas2...]
  """

  out = None
  bag = rosbag.Bag(ros_bag_file)
  out = collections.defaultdict(list)
  if topics is None:
    topics = []
  for topic, msg, _ in bag.read_messages(topics=topics):
    out[topic.split('/')[-1]].append(msg)
  bag.close()
  logger.info('Number of topics: {}'.format(len(out)))
  return out

def extract_gps_imu_data(gps_imu_list):
  """Convert gps_imu_data to dict.

  For now, we only extract pos_type and status.
  """
  gps_imu_list.sort(key=lambda x: x.timestamp)
  gps_imu_dict = {}
  for val in gps_imu_list:
    gps_imu_dict[val.timestamp] = {}
    # Get pos_type and status by enum value.
    descriptor = val.DESCRIPTOR
    pos_type = descriptor.fields_by_name['pos_type'].enum_type.values_by_number[
        val.pos_type].name
    status = descriptor.fields_by_name['status'].enum_type.values_by_number[
        val.status].name
    # Get position covariance.
    try:
      if len(val.position_covariance) == 9:
        x_ind, y_ind, z_ind = 0, 4, 8
      elif len(val.position_covariance) == 3:
        x_ind, y_ind, z_ind = 0, 1, 2
      gps_imu_dict[val.timestamp]['lat_cov'] = val.position_covariance[x_ind]
      gps_imu_dict[val.timestamp]['long_cov'] = val.position_covariance[y_ind]
      gps_imu_dict[val.timestamp]['alt_cov'] = val.position_covariance[z_ind]
    except (IndexError, UnboundLocalError):
      gps_imu_dict[val.timestamp]['lat_cov'] = INVALID_VARIANCE
      gps_imu_dict[val.timestamp]['long_cov'] = INVALID_VARIANCE
      gps_imu_dict[val.timestamp]['alt_cov'] = INVALID_VARIANCE

    gps_imu_dict[val.timestamp]['pos_type'] = pos_type
    gps_imu_dict[val.timestamp]['status'] = status
  return gps_imu_dict

def pose_to_dict(proto_list):
  """Convert pose to dict."""
  out = {}
  out['t'] = [val.timestamp for val in proto_list]
  out['lat'] = [val.latitude for val in proto_list]
  out['lon'] = [val.longitude for val in proto_list]
  out['x'], out['y'] = lat_lon_list_to_x_y(out['lat'], out['lon'])
  out['z'] = [val.z for val in proto_list]
  out['vel_x'] = [val.vel_x for val in proto_list]
  out['vel_y'] = [val.vel_y for val in proto_list]
  out['vel_z'] = [val.vel_z for val in proto_list]
  out['acc_right'] = [val.acc_right for val in proto_list]
  out['acc_forward'] = [val.acc_forward for val in proto_list]
  out['acc_up'] = [val.acc_up for val in proto_list]
  out['roll'] = [val.roll for val in proto_list]
  out['pitch'] = [val.pitch for val in proto_list]
  out['yaw'] = [val.yaw for val in proto_list]
  out['x_var'] = []
  out['y_var'] = []
  out['z_var'] = []
  for val in proto_list:
    try:
      if len(val.position_covariance) == 9:
        x_ind, y_ind, z_ind = 0, 4, 8
      elif len(val.position_covariance) == 3:
        x_ind, y_ind, z_ind = 0, 1, 2
      out['x_var'].append(val.position_covariance[x_ind])
      out['y_var'].append(val.position_covariance[y_ind])
      out['z_var'].append(val.position_covariance[z_ind])
    except (IndexError, UnboundLocalError):
      out['x_var'].append(INVALID_VARIANCE)
      out['y_var'].append(INVALID_VARIANCE)
      out['z_var'].append(INVALID_VARIANCE)
  return out

def ins_pose_to_dict(proto_list, gps_imu_dict):
  """Convert ins poses to dict."""
  out_dict = {}
  out_dict = pose_to_dict(proto_list)

  if gps_imu_dict:
    metrics = ['pos_type', 'status', 'lat_cov', 'long_cov', 'alt_cov']
    for metric in metrics:
      out_dict[metric] = []
    for t in out_dict['t']:
      if t in gps_imu_dict.keys():
        for metric in metrics:
          out_dict[metric].append(gps_imu_dict[t][metric])

  return out_dict

def write_gps_imu_stats(gps_imu_dict, json_dir):
  out_dict = {}
  out_dict['count'] = len(gps_imu_dict)

  pos_type_dict = collections.defaultdict(int)
  status_dict = collections.defaultdict(int)
  for t in gps_imu_dict:
    pos_type_dict[gps_imu_dict[t]["pos_type"]] += 1
    status_dict[gps_imu_dict[t]["status"]] += 1
  out_dict['pos_type'] = pos_type_dict
  out_dict['status'] = status_dict

  gps_imu_status_json_file = os.path.join(json_dir, 'gps_imu_stats.json')
  with open(gps_imu_status_json_file, 'w') as outfile:
    json.dump(out_dict, outfile)

def write_ins_pose(gps_imu_list, ins_pose_list, json_dir):
  """Dump ins poses into json output file."""
  gps_imu_dict = {}
  if gps_imu_list:
    gps_imu_dict = extract_gps_imu_data(gps_imu_list)

  ins_pose_list = list(ins_pose_list)
  ins_pose_list.sort(key=lambda x: x.timestamp)
  ins_pose_dict = ins_pose_to_dict(ins_pose_list, gps_imu_dict)
  pose_json_file = os.path.join(json_dir, 'ins_pose.json')
  with open(pose_json_file, 'w') as outfile:
    json.dump(ins_pose_dict, outfile)

  if gps_imu_dict:
    write_gps_imu_stats(gps_imu_dict, json_dir)


def write_pose(proto_list, json_dir, pose_name):
  """Dump car poses into json output file."""
  # Sort the list by timestamp.
  proto_list = list(proto_list)
  proto_list.sort(key=lambda x: x.timestamp)

  pose_dict = {}
  pose_dict = pose_to_dict(proto_list)

  pose_json_file = os.path.join(json_dir, pose_name + '.json')
  with open(pose_json_file, 'w') as outfile:
    json.dump(pose_dict, outfile)


def euler_angles_to_quaternion(roll, pitch, yaw):
    '''
    Convert roll, pitch, yaw to quaternion
    '''
    qx = Quaternion(axis=[1., 0., 0.], angle=pitch)
    qy = Quaternion(axis=[0., 1., 0.], angle=roll)
    qz = Quaternion(axis=[0., 0., 1.], angle=yaw)
    return qz * qy * qx

def run(trip_id, rosbag_file, out_folder):
  dest_folder = os.path.join(out_folder, trip_id)
  if os.path.exists(dest_folder):
    shutil.rmtree(dest_folder)

  # Copy dashboard template
  dashboard_template_dir = os.path.join(os.getcwd(), 'dashboard')
  shutil.copytree(dashboard_template_dir, dest_folder)

  # Key: name of the pose, Value: poses in pose.proto
  pose_dict = read_topics_from_ros_bag(rosbag_file,
                                         topics=[
                                         '/raw_imu_data',
                                         '/gps_imu_data',
                                         '/localization/ins_pose',
                                         '/pose'])

  # Dump imu_data to CSV in Euroc format.
  print("Dump imu data starting")
  def extract_imu_data(raw_imu_list):
    raw_imu_list.sort(key=lambda x: x.timestamp)
    imu_data = []
    for val in raw_imu_list:
      timestamp = val.timestamp * 1000000
      velocity_y = val.velocity_pitch
      velocity_z = val.velocity_roll
      velocity_x = val.velocity_yaw
      acceleration_y = val.acceleration_right
      acceleration_z = val.acceleration_forward
      acceleration_x = val.acceleration_up

      imu_data.append([timestamp, velocity_x, velocity_y, velocity_z, acceleration_x, acceleration_y, acceleration_z])
    return imu_data

  csv_dir = os.path.join(dest_folder, 'csv')
  if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)
  imu_data_file = os.path.join(csv_dir, 'imu_data.csv')
  imu_data = extract_imu_data(pose_dict['raw_imu_data'])
  with open(imu_data_file,'w') as csv_file:
    writer = csv.writer(csv_file)
    [writer.writerow(item) for item in imu_data]


  # Dump ground truth to CSV in Euroc format.
  print("Dump imu data starting")
  def extract_gt_data(gps_imu_list):
    gps_imu_list.sort(key=lambda x: x.timestamp)
    gt_data = []
    for val in gps_imu_list:
      timestamp = val.timestamp * 1000000
      utm_x, utm_y, _, _ = utm.from_latlon(val.latitude,val.longitude)
      altitude = val.altitude
      pitch = val.pitch
      roll = val.roll
      yaw = val.yaw
      q = euler_angles_to_quaternion(roll, pitch, yaw)
      q_x = q.elements
      # velocity_east = val.velocity_east
      # velocity_north = val.velocity_north
      # velocity_up = val.velocity_up
      velocity_y = val.velocity_east
      velocity_z = val.velocity_north
      velocity_x = val.velocity_up

      gt_data.append([timestamp, utm_x, utm_y, altitude, q_x[0], q_x[1], q_x[2], q_x[3], velocity_x, velocity_y, velocity_z])
    return gt_data

  csv_dir = os.path.join(dest_folder, 'csv')
  if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)
  gt_file = os.path.join(csv_dir, 'ground_truth.csv')
  gt_data = extract_gt_data(pose_dict['gps_imu_data'])
  with open(gt_file,'w') as csv_file:
    writer = csv.writer(csv_file)
    [writer.writerow(item) for item in gt_data]

  # Create json folder
  json_dir = os.path.join(dest_folder, 'json')
  if not os.path.exists(json_dir):
    os.makedirs(json_dir)
  
  print("Visualizing poses.")

  # Dump gps_imu_data if exists.
  # Visualize ins pose.
  print("Visualizing ins_pose")
  write_ins_pose(pose_dict['gps_imu_data'],
                          pose_dict['ins_pose'],
                          json_dir)
  # Save poses to json file for display in satellite view.
  write_pose(pose_dict['pose'], json_dir, "pose")

  print("Visualization complete, please check {0}".format(
      os.path.join(dest_folder, 'dashboard.html')))


def main():
  """main function"""
  parser = OptionParser()
  parser.add_option('-t', '--trip_id')
  parser.add_option('-r', '--rosbag_file')
  parser.add_option('-o', '--out_folder')
  options, _ = parser.parse_args(sys.argv)
  run(options.trip_id, options.rosbag_file, options.out_folder)


if __name__ == "__main__":
  main()
