/*
 * Created on Sun Sep 12 2021
 *
 * Copyright (c) 2021 HITsz-NRSL
 *
 * Author: EpsAvlc
 */

#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/gicp.h>
#include <sensor_msgs/PointCloud2.h>
#include <ros/package.h>

#include "flexible_icp/flexible_icp.h"

#ifndef PCL_NO_PRECOMPILE
#include <pcl/impl/instantiate.hpp>
#include <pcl/search/impl/search.hpp>
PCL_INSTANTIATE(Search, PCL_POINT_TYPES)

#endif  // PCL_NO_PRECOMPILE

int main(int argc, char **argv) {
  ros::init(argc, argv, "flexible_icp_test");
  ros::NodeHandle nh;

  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>),
      target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::io::loadPCDFile(ros::package::getPath("flexible_icp") +  "/data/251370668.pcd", *source_cloud);
  pcl::io::loadPCDFile(ros::package::getPath("flexible_icp") + "/data/251371071.pcd", *target_cloud);

  pcl::FlexibleICP ficp(pcl::FlexibleICP::POINT_TO_PLANE, "000111");
  ficp.setInputSource(source_cloud);
  ficp.setInputTarget(target_cloud);
  ficp.setMaxCorrespondenceDistance(5);
  pcl::PointCloud<pcl::PointXYZ> aligned_cloud;
  ros::Time t0 = ros::Time::now();
  ficp.align(aligned_cloud);
  std::cout << "Elapse " << (ros::Time::now() - t0).toSec() << std::endl;
  std::cout << ficp.getFinalTransformation() << std::endl;
  std::cout << "finish align." << std::endl;

  /* visualization */
  ros::Publisher source_pub = nh.advertise<sensor_msgs::PointCloud2>("source_cloud", 1);
  ros::Publisher target_pub = nh.advertise<sensor_msgs::PointCloud2>("target_cloud", 1);
  ros::Publisher aligned_pub = nh.advertise<sensor_msgs::PointCloud2>("aligned_cloud", 1);
  sensor_msgs::PointCloud2 source_msg, target_msg, aligned_msg;
  pcl::toROSMsg(*source_cloud, source_msg);
  pcl::toROSMsg(*target_cloud, target_msg);
  pcl::toROSMsg(aligned_cloud, aligned_msg);

  source_msg.header.frame_id = target_msg.header.frame_id = aligned_msg.header.frame_id = "map";

  ros::Rate loop(3);
  while (ros::ok()) {
    source_msg.header.stamp = target_msg.header.stamp = aligned_msg.header.stamp = ros::Time::now();
    source_pub.publish(source_msg);
    target_pub.publish(target_msg);
    aligned_pub.publish(aligned_msg);
    loop.sleep();
  }
}
