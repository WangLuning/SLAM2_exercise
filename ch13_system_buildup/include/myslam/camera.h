#pragma once
#ifndef MYSLAM_CAMERA_H
#define MYSLAM_CAMERA_H

#include "common_include.h"

// extrinsics: where the camera is located and facing, i.e. pose
//			6 dim
// intrinsics: how the camera inside is built
//			4 / 5 dim of freedom whether a digital camera or not
// usually total 11 dims of freedom in Direct Linear Transformation (DLT) with no lens distortion
// another additionally added nonlinear parameters like distortion


// coordinate systems
// 1. world: the position of projection center X_o, rotation R
// 2. camera: the origin is the projection center
// 3. image plane
// 4. sensor
namespace myslam {
	// define a pinhole stereo camera
	class Camera {
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		typedef std::shared_ptr<Camera> Ptr;

		// intrinsic parameters
		double fx_ = 0, fy_ = 0, cx_ = 0, cy_ = 0, baseline_ = 0;
		// extrinsic, from stereo to single
		// the way to transform between two poses
		SE3 pose_;
		SE3 pose_inv_;

		Camera();
		Camera(double fx, double fy, double cx, double cy, double baseline, const SE3& pose):
			fx_(fx), fy_(fy), cx_(cx), cy_(cy), baseline_(baseline), pose_(pose)
		{
			pose_inv_ = pose_.inverse();
		}

		SE3 pose() const { return pose_; }

		Mat33 K() const {
			Mat33 k;
			k << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1;
			return k;
		}

		// transformation between 3 set of coordinates
		// world, camera, pixel
		Vec3 world2camera(const Vec3& p_w, const SE3& T_c_w);
		Vec3 camera2world(const Vec3& p_c, const SE3& T_c_w);
		Vec2 camera2pixel(const Vec3& p_c);
		Vec3 pixel2camera(const Vec2& p_p, double depth = 1);
		Vec3 pixel2world(const Vec2& p_p, const SE3& T_c_w, double depth = 1);
		Vec2 world2pixel(const Vec3& p_w, const SE3& T_c_w);
	};
}
#endif // !MYSLAM_CAMERA_H
