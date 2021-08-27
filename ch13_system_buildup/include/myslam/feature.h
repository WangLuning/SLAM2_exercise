#pragma once
#ifndef MYSLAM_FEATURE_H
#define MYSLAM_FEATURE_H

#include <memory>
#include <opencv2/features2d.hpp>
#include "common_include.h"

namespace myslam {
	struct Frame;
	struct MapPoint;

	// feature points in 2D
	// after triangulation it will be mapped to a map point
	struct Feature {
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		typedef std::shared_ptr<Feature> Ptr;

		// which frame is being used
		std::weak_ptr<Frame> frame_;
		// key points extracted from the frame
		cv::KeyPoint position_;
		// the corresponding map point
		std::weak_ptr<MapPoint> map_point_;

		// if this is an outlier (too far away etc.) remove it
		bool is_outlier_ = false;
		bool is_on_left_image_ = true;

	public:
		Feature() {}
		Feature(std::shared_ptr<Frame>frame, const cv::KeyPoint& kp) :
			frame_(frame), position_(kp) {}

	};
}
#endif // !MYSLAM_FEATURE_H
