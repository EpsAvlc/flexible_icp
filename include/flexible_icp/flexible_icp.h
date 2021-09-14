/*
 * Created on Sat Sep 11 2021
 *
 * Copyright (c) 2021 HITsz-NRSL
 *
 * Author: EpsAvlc
 */

#pragma once

#include <pcl/registration/registration.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Core>
#include <vector>
#include <string>

namespace pcl {
using PointSource = pcl::PointXYZ;
using PointTarget = pcl::PointXYZ;
class FlexibleICP : public Registration<PointSource, PointTarget> {
 public:
  using Vector6f = Eigen::Matrix<float, 6, 1>;
  using Matrix6f = Eigen::Matrix<float, 6, 6>;
  using Ptr = boost::shared_ptr<FlexibleICP>;
  enum Method {
    STANDARD,
    POINT_TO_PLANE,
    GENERALIZED,
  };
  explicit FlexibleICP(const Method &m, int optimize_flag);
  explicit FlexibleICP(const Method &m, std::string optimize_flag);

  void setInputSource(const PointCloudSourceConstPtr &cloud) override;

  void setInputTarget(const PointCloudTargetConstPtr &cloud) override;

  void computeTransformation(PointCloudSource &output, const Matrix4 &guess) override;

 private:
  void initParameters();

  void calculateTargetNormals();

  void estimateTransformation(const pcl::PointCloud<PointSource> &cloud_src,
                                      const std::vector<int> &src_indices,
                                      const pcl::PointCloud<PointTarget> &cloud_tgt,
                                      const std::vector<int> &tgt_indices, Eigen::Matrix4f *transformation_matrix);

  void linearizedStandard(const pcl::PointCloud<PointSource> &cloud_src, const std::vector<int> &src_indices,
                          const pcl::PointCloud<PointTarget> &cloud_tgt, const std::vector<int> &tgt_indices,
                          const Vector6f &x, Eigen::MatrixXf *H, Eigen::VectorXf *g, float* chi);

  void linearizedPointToPlane(const pcl::PointCloud<PointSource> &cloud_src, const std::vector<int> &src_indices,
                          const pcl::PointCloud<PointTarget> &cloud_tgt, const std::vector<int> &tgt_indices,
                          const Vector6f &x, Eigen::MatrixXf *H, Eigen::VectorXf *g, float* chi);

  bool isStateConverged(const Eigen::MatrixXf &dx);

  Vector6f oPlus(const Vector6f &x, Eigen::VectorXf dx);


  Method method_;
  uint8_t optimize_flag_; /*x, y, z, r, p, y*/
  uint8_t state_size_;
  std::vector<uint8_t> state_map_;
  pcl::search::KdTree<PointTarget>::Ptr kdtree_target_;
  int max_inner_iteration_num_;
  int max_outer_iteration_num_;
  float inner_iteration_threshold_;
  double translation_epsilon_;
  double rotation_epsilon_;

  pcl::PointCloud<pcl::Normal> normals_target_;
  /* PointToPlane */
};
}  // namespace pcl
