#pragma once
#ifndef  MYSLAM_FRAME_H
#define MYSLAM_FRAME_H

#include "camera.h"
#include "common_include.h"

namespace myslam {
	struct MapPoint;
	struct Feature;

	// here each frame can be a pic from monocular camera
	// or a set of two pics from a binocular camera
	struct Frame {
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		typedef std::shared_ptr<Frame> Ptr;

		unsigned long id_ = 0;
		unsigned long keyframe_id_ = 0;
		// we only track a few as key frames to minimize computation
		bool is_keyframe_ = false;
		double time_stamp_;
		SE3 pose_;
		std::mutex pose_mutex_;
		// stereo images, can be mono or bino
		cv::Mat left_img_, right_img_;


		// extract features in the left image
		std::vector<std::shared_ptr<Feature>> features_left_;
		// corresponding features in the right image
		// but we do not necessarily have one
		std::vector<std::shared_ptr<Feature>> features_right_;

	public:
		Frame(){}
		Frame(long id, double time_stamp, const SE3& pose, const Mat& left, const Mat& right);

		SE3 Pose() {
			std::unique_lock<std::mutex> lck(pose_mutex_);
			return pose_;
		}

		void SetPose(const SE3& pose) {
			std::unique_lock<std::mutex> lck(pose_mutex_);
			pose_ = pose;
		}

		// most are normal frames
		void SetKeyFrame();

		// factory mode, assign id
		static std::shared_ptr<Frame> CreateFrame();
	};
}
#endif // ! MYSLAM_FRAME_H
