/*
 * Created on Sat Sep 11 2021
 *
 * Copyright (c) 2021 HITsz-NRSL
 *
 * Author: EpsAvlc
 */

#include "flexible_icp/flexible_icp.h"

#include <Eigen/Dense>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/console/time.h>

#include <omp.h>

namespace pcl {

FlexibleICP::FlexibleICP(const Method &m, int optimize_flag)
    : Registration(), method_(m), optimize_flag_(optimize_flag) {
  initParameters();
}

FlexibleICP::FlexibleICP(const Method &m, std::string optimize_flag)
    : Registration(), method_(m), max_inner_iteration_num_(100), max_outer_iteration_num_(10), rotation_epsilon_(2e-3),
      translation_epsilon_(5e-4) {
  optimize_flag_ = std::stoi(optimize_flag, nullptr, 2);
  for (int i = 0; i < optimize_flag.size(); ++i) {
    if (optimize_flag[i] == '1') {
      state_map_.push_back(i);
    }
  }
  state_size_ = state_map_.size();
  initParameters();
}

void FlexibleICP::initParameters() {
  max_inner_iteration_num_ = 100;
  max_outer_iteration_num_ = 100;
  rotation_epsilon_ = 2e-3;
  translation_epsilon_ = 5e-4;
  corr_dist_threshold_ = 5;
  kdtree_target_.reset(new pcl::search::KdTree<PointTarget>);
}

void FlexibleICP::calculateTargetNormals() {
  normals_target_.clear();
  pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
  ne.setInputCloud(target_);
  ne.setSearchMethod(kdtree_target_);
  ne.setKSearch(15);
  ne.compute(normals_target_);
}

void FlexibleICP::setInputSource(const PointCloudSourceConstPtr &cloud) {
  Registration<PointSource, PointTarget>::setInputSource(cloud);
  switch (method_) {
  case POINT_TO_PLANE:
    /* code */
    break;
  case GENERALIZED:
    break;
  default:
    break;
  }
}

void FlexibleICP::setInputTarget(const PointCloudTargetConstPtr &cloud) {
  Registration<PointSource, PointTarget>::setInputTarget(cloud);
  kdtree_target_->setInputCloud(cloud);
  pcl::console::TicToc tt;
  switch (method_) {
  case POINT_TO_PLANE:
    tt.tic();
    calculateTargetNormals();
    PCL_INFO("[calculateTargetNormals] elapse: %f", tt.toc());
    break;
  case GENERALIZED:
    break;
  default:
    break;
  }
}

void FlexibleICP::computeTransformation(PointCloudSource &output, const Matrix4 &guess) {
  PointCloudSourcePtr input_transformed(new PointCloudSource);

  final_transformation_ = guess;
  if (guess != Matrix4::Identity()) {
    pcl::transformPointCloud(output, output, guess);
  }

  transformation_ = Matrix4::Identity();

  converged_ = false;
  Matrix4 transform_R = Matrix4::Zero();
  int N = output.size();
  for (int iter = 0; iter < max_outer_iteration_num_; ++iter) {
    transform_R = transformation_ * guess;

    int cnt = 0;
    std::vector<int> source_indices(indices_->size());
    std::vector<int> target_indices(indices_->size());
    std::vector<std::vector<int>> nn_indices_vec(omp_get_max_threads());
    std::vector<std::vector<float>> nn_dists_vec(omp_get_max_threads());
#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
      std::vector<int> &nn_indices = nn_indices_vec[omp_get_thread_num()];
      std::vector<float> &nn_dists = nn_dists_vec[omp_get_thread_num()];
      PointSource query = output[i];
      query.getVector4fMap() = transformation_ * query.getVector4fMap();

      if (!tree_->nearestKSearch(query, 2, nn_indices, nn_dists)) {
        PCL_ERROR("[pcl::%s::computeTransformation] Unable to find a nearest "
                  "neighbor in the target dataset for point %d in the source!\n",
                  getClassName().c_str(), (*indices_)[i]);
        continue;
      }
      if (nn_dists[0] < corr_dist_threshold_) {
#pragma omp critical
        {
          source_indices[cnt] = static_cast<int>(i);
          target_indices[cnt] = nn_indices[0];
          ++cnt;
        }
      }
    }

    source_indices.resize(cnt);
    target_indices.resize(cnt);

    previous_transformation_ = transformation_;
    estimateTransformation(output, source_indices, *target_, target_indices, &transformation_);

    double delta = 0.;
    for (int k = 0; k < 4; k++) {
      for (int l = 0; l < 4; l++) {
        double ratio = 1;
        if (k < 3 && l < 3)  // rotation part of the transform
          ratio = 1. / rotation_epsilon_;
        else
          ratio = 1. / translation_epsilon_;
        double c_delta = ratio * std::abs(previous_transformation_(k, l) - transformation_(k, l));
        if (c_delta > delta)
          delta = c_delta;
      }
    }

    if (delta < 1) {
      break;
    }
  }
  final_transformation_ = previous_transformation_ * guess;

  // Transform the point cloud
  pcl::transformPointCloud(*input_, output, final_transformation_);
}

void FlexibleICP::estimateTransformation(const pcl::PointCloud<PointSource> &cloud_src,
                                         const std::vector<int> &src_indices,
                                         const pcl::PointCloud<PointTarget> &cloud_tgt,
                                         const std::vector<int> &tgt_indices, Eigen::Matrix4f *transformation_matrix) {
  if (src_indices.size() != tgt_indices.size()) {
    PCL_ERROR("[pcl::%s::estunateTrabsformationStandard]: source indices has different size %i with target size %i!\n",
              src_indices.size(), tgt_indices.size());
    return;
  }
  Vector6f x = Vector6f::Zero();
  x[0] = (*transformation_matrix)(0, 3);
  x[1] = (*transformation_matrix)(1, 3);
  x[2] = (*transformation_matrix)(2, 3);
  x[3] = std::atan2((*transformation_matrix)(2, 1), (*transformation_matrix)(2, 2));  // roll
  x[4] = asin(-(*transformation_matrix)(2, 0));                                       // pitch
  x[5] = std::atan2((*transformation_matrix)(1, 0), (*transformation_matrix)(0, 0));  // yaw

  /* paramters of Levenberg-Marquardt algorithm */
  // see https://people.duke.edu/~hpgavin/ce281/lm.pdf
  float lambda = -1;
  float rho = 0;
  float rho_epsilon = 0.75;
  float vi = 2;
  Eigen::MatrixXf H;
  Eigen::VectorXf g;
  float chi = 0;

  switch (method_) {
  case STANDARD:
    linearizedStandard(cloud_src, src_indices, cloud_tgt, tgt_indices, x, &H, &g, &chi);
    break;
  case POINT_TO_PLANE:
    linearizedPointToPlane(cloud_src, src_indices, cloud_tgt, tgt_indices, x, &H, &g, &chi);
    break;
  default:
    break;
  }

  /* Initialization and update of the L-M parameter, λ*/
  float max_diag = std::numeric_limits<float>::min();
  if (lambda < 0) {
    for (int i = 0; i < state_size_; ++i) {
      max_diag = std::max(max_diag, H(i, i));
    }
    lambda = max_diag * 0.1;
  }
  bool inner_converged = false;
  int iter_num = 0;
  while (iter_num < max_inner_iteration_num_) {
    Eigen::HouseholderQR<Eigen::MatrixXf> qr(H + lambda * Eigen::MatrixXf::Identity(state_size_, state_size_));
    Eigen::VectorXf dx = qr.solve(g);
    // std::cout << "H: " << H << std::endl;
    // std::cout << "g: " << g << std::endl;
    // std::cout << "dx: " << dx << std::endl;
    if (isStateConverged(dx)) {
      inner_converged = true;
      oPlus(x, dx);
      break;
    }

    Vector6f x_1 = oPlus(x, dx);
    float chi_1 = 0;
    Eigen::VectorXf new_g;
    Eigen::MatrixXf new_H;
    switch (method_) {
    case STANDARD:
      linearizedStandard(cloud_src, src_indices, cloud_tgt, tgt_indices, x_1, &new_H, &new_g, &chi_1);
      break;
    case POINT_TO_PLANE:
      linearizedPointToPlane(cloud_src, src_indices, cloud_tgt, tgt_indices, x_1, &new_H, &new_g, &chi_1);
    default:
      break;
    }

    rho = (chi - chi_1) / (dx.transpose() * (lambda * dx + g));
    if (rho <= rho_epsilon) {
      lambda *= vi;
      vi = 2 * vi;
    } else {
      int j = 0;
      x = oPlus(x, dx);
      lambda = lambda * std::max(1.0 / 3.0, 1 - std::pow((2 * rho - 1), 3));
      vi = 2;
      H = new_H;
      g = new_g;
    }
  }

  (*transformation_matrix)(0, 3) = x[0];
  (*transformation_matrix)(1, 3) = x[1];
  (*transformation_matrix)(2, 3) = x[2];

  Eigen::Matrix3f Rx = Eigen::AngleAxisf(x[3], Eigen::Vector3f::UnitX()).matrix();
  Eigen::Matrix3f Ry = Eigen::AngleAxisf(x[4], Eigen::Vector3f::UnitY()).matrix();
  Eigen::Matrix3f Rz = Eigen::AngleAxisf(x[5], Eigen::Vector3f::UnitZ()).matrix();

  transformation_matrix->block(0, 0, 3, 3) = Rz * Ry * Rx;
}

void FlexibleICP::linearizedStandard(const pcl::PointCloud<PointSource> &cloud_src, const std::vector<int> &src_indices,
                                     const pcl::PointCloud<PointTarget> &cloud_tgt, const std::vector<int> &tgt_indices,
                                     const Vector6f &x, Eigen::MatrixXf *H, Eigen::VectorXf *g, float *chi) {
  if (nullptr != H) {
    H->setZero(state_size_, state_size_);
  }
  if (nullptr != g) {
    g->setZero(state_size_, 1);
  }
  if (nullptr != chi) {
    *chi = 0;
  }

  Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
  Eigen::Matrix3f pitch_J_mat, roll_J_mat, yaw_J_mat;
  pitch_J_mat = roll_J_mat = yaw_J_mat = Eigen::Matrix3f::Zero();
  float sr = sin(x[3]), cr = cos(x[3]), sp = sin(x[4]), cp = cos(x[4]), sy = sin(x[5]), cy = cos(x[5]);
  Eigen::Matrix3f Rx = Eigen::AngleAxisf(x[3], Eigen::Vector3f::UnitX()).matrix();
  Eigen::Matrix3f Ry = Eigen::AngleAxisf(x[4], Eigen::Vector3f::UnitY()).matrix();
  Eigen::Matrix3f Rz = Eigen::AngleAxisf(x[5], Eigen::Vector3f::UnitZ()).matrix();
  R = Rz * Ry * Rx;

  roll_J_mat(1, 1) = -sr;
  roll_J_mat(1, 2) = -cr;
  roll_J_mat(2, 1) = cr;
  roll_J_mat(2, 2) = -sr;

  pitch_J_mat(0, 0) = -sp;
  pitch_J_mat(0, 2) = cp;
  pitch_J_mat(2, 0) = -cp;
  pitch_J_mat(2, 2) = -sp;

  yaw_J_mat(0, 0) = -sy;
  yaw_J_mat(0, 1) = -cy;
  yaw_J_mat(1, 0) = cy;
  yaw_J_mat(1, 1) = -sy;

  Eigen::Matrix3f J3, J4, J5;
  J3 = Rz * Ry * roll_J_mat;
  J4 = Rz * pitch_J_mat * Rx;
  J5 = yaw_J_mat * Ry * Rz;

  Eigen::Vector3f t = x.head<3>();
  for (int i = 0; i < src_indices.size(); ++i) {
    const Eigen::Vector3f &pt_src = cloud_src[src_indices[i]].getVector3fMap();
    const Eigen::Vector3f &pt_tgt = cloud_tgt[tgt_indices[i]].getVector3fMap();
    Eigen::MatrixXf J = Eigen::MatrixXf::Zero(3, state_size_);
    for (int j = 0; j < state_map_.size(); ++j) {
      switch (state_map_[j]) {
      case 0:
        J.col(j) = Eigen::Vector3f::UnitX();
        break;
      case 1:
        J.col(j) = Eigen::Vector3f::UnitY();
        break;
      case 2:
        J.col(j) = Eigen::Vector3f::UnitZ();
        break;
      case 3:
        J.col(j) = J3 * pt_src;
        break;
      case 4:
        J.col(j) = J4 * pt_src;
        break;
      case 5:
        J.col(j) = J5 * pt_src;
        break;
      default:
        break;
      }
    }

    Eigen::Vector3f residual = R * pt_src + t - pt_tgt;
    if (nullptr != H) {
      (*H) += J.transpose() * J;
    }
    if (nullptr != g) {
      (*g) += -J.transpose() * residual;
    }
    if (nullptr != chi) {
      (*chi) += residual.squaredNorm();
    }
  }
  (*H) /= src_indices.size();
  (*g) /= src_indices.size();
  (*chi) /= src_indices.size();
}

void FlexibleICP::linearizedPointToPlane(const pcl::PointCloud<PointSource> &cloud_src,
                                         const std::vector<int> &src_indices,
                                         const pcl::PointCloud<PointTarget> &cloud_tgt,
                                         const std::vector<int> &tgt_indices, const Vector6f &x, Eigen::MatrixXf *H,
                                         Eigen::VectorXf *g, float *chi) {
  if (nullptr != H) {
    H->setZero(state_size_, state_size_);
  }
  if (nullptr != g) {
    g->setZero(state_size_, 1);
  }
  if (nullptr != chi) {
    *chi = 0;
  }

  Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
  Eigen::Matrix3f pitch_J_mat, roll_J_mat, yaw_J_mat;
  pitch_J_mat = roll_J_mat = yaw_J_mat = Eigen::Matrix3f::Zero();
  float sr = sin(x[3]), cr = cos(x[3]), sp = sin(x[4]), cp = cos(x[4]), sy = sin(x[5]), cy = cos(x[5]);
  Eigen::Matrix3f Rx = Eigen::AngleAxisf(x[3], Eigen::Vector3f::UnitX()).matrix();
  Eigen::Matrix3f Ry = Eigen::AngleAxisf(x[4], Eigen::Vector3f::UnitY()).matrix();
  Eigen::Matrix3f Rz = Eigen::AngleAxisf(x[5], Eigen::Vector3f::UnitZ()).matrix();
  R = Rz * Ry * Rx;

  roll_J_mat(1, 1) = -sr;
  roll_J_mat(1, 2) = -cr;
  roll_J_mat(2, 1) = cr;
  roll_J_mat(2, 2) = -sr;

  pitch_J_mat(0, 0) = -sp;
  pitch_J_mat(0, 2) = cp;
  pitch_J_mat(2, 0) = -cp;
  pitch_J_mat(2, 2) = -sp;

  yaw_J_mat(0, 0) = -sy;
  yaw_J_mat(0, 1) = -cy;
  yaw_J_mat(1, 0) = cy;
  yaw_J_mat(1, 1) = -sy;

  Eigen::Matrix3f J3, J4, J5;
  J3 = Rz * Ry * roll_J_mat;
  J4 = Rz * pitch_J_mat * Rx;
  J5 = yaw_J_mat * Ry * Rz;

  Eigen::Vector3f t = x.head<3>();
  // #pragma omp parallel for
  for (int i = 0; i < src_indices.size(); ++i) {
    const Eigen::Vector3f &pt_src = cloud_src[src_indices[i]].getVector3fMap();
    const Eigen::Vector3f &pt_tgt = cloud_tgt[tgt_indices[i]].getVector3fMap();
    const Eigen::Vector3f &n_tgt = normals_target_[tgt_indices[i]].getNormalVector3fMap();
    if (!pcl::isFinite(normals_target_[tgt_indices[i]])) {
      continue;
    }
    Eigen::MatrixXf J = Eigen::MatrixXf::Zero(1, state_size_);
    for (int j = 0; j < state_map_.size(); ++j) {
      switch (state_map_[j]) {
      case 0:
        J.col(j) = n_tgt.transpose() * Eigen::Vector3f::UnitX();
        break;
      case 1:
        J.col(j) = n_tgt.transpose() * Eigen::Vector3f::UnitY();
        break;
      case 2:
        J.col(j) = n_tgt.transpose() * Eigen::Vector3f::UnitZ();
        break;
      case 3:
        J.col(j) = n_tgt.transpose() * J3 * pt_src;
        break;
      case 4:
        J.col(j) = n_tgt.transpose() * J4 * pt_src;
        break;
      case 5:
        J.col(j) = n_tgt.transpose() * J5 * pt_src;
        break;
      default:
        break;
      }
    }
    Eigen::Matrix<float, 1, 1> residual = n_tgt.transpose() * (R * pt_src + t - pt_tgt);
    // #pragma omp critical
    {
      if (nullptr != H) {
        (*H) += J.transpose() * J;
        // std::cout << "H: " << *H << std::endl;
      }
      if (nullptr != g) {
        (*g) += -J.transpose() * residual;
        // std::cout << "g: " << *g << std::endl;
      }
      if (nullptr != chi) {
        (*chi) += residual.squaredNorm();
      }
    }
  }
  (*H) /= src_indices.size();
  (*g) /= src_indices.size();
  (*chi) /= src_indices.size();
}

bool FlexibleICP::isStateConverged(const Eigen::MatrixXf &dx) {
  for (int i = 0; i < state_size_; ++i) {
    if (fabs(dx(i)) > translation_epsilon_ / 10) {
      return false;
    }
  }
  return true;
}

FlexibleICP::Vector6f FlexibleICP::oPlus(const Vector6f &x, Eigen::VectorXf dx) {
  Vector6f res = x;
  int cnt = 0;
  for (int i = 0; i < state_map_.size(); ++i) {
    res[state_map_[i]] += dx[i];
  }
  return res;
}

}  // namespace pcl
