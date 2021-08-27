#pragma once
#ifndef MYSLAM_FRONTEND_H
#define MYSLAM_FRONTEND_H

// frontend needs to detect and calculate feature matching
// backend is for optimizing the whole problem

#include <opencv2/features2d.hpp>
#include "common_include.h"
#include "frame.h"
#include "map.h"

namespace myslam {
	class Backend;
	class Viewer;

	enum class FrontendStatus{INITING, TRACKING_GOOD, TRACKING_BAD, LOST};

	class Frontend {
	private:
		// when normal
		bool Track();

		// when lost
		bool Reset();

		//track with last frame, return num of tracked points
		int TrackLastFrame();

		// estimate cur frame pose, return num of inliers
		int EstimateCurrentPose();

		//set cur as key frame and insert it into backend
		bool InsertKeyframe();

		// init frontend with stereo images
		bool StereoInit();

		//detect features in left image in current_frame_
		int DetectFeatures();

		//find corresponding features in right
		int FindFeaturesInRight();

		// build init map with single image
		bool BuildInitMap();

		// triangulation with 2d points, return num of triangulated points
		int TriangulateNewPoints();

		// set the features in keyframe as new observation of map points
		void SetObservationsForKeyFrame();

		FrontendStatus status_ = FrontendStatus::INITING;

		Frame::Ptr current_frame_ = nullptr;
		Frame::Ptr last_frame_ = nullptr;
		Camera::Ptr camera_left_ = nullptr;
		Camera::Ptr camera_right_ = nullptr;

		Map::Ptr map_ = nullptr;
		std::shared_ptr<Backend> backend_ = nullptr;
		std::shared_ptr<Viewer> viewer_ = nullptr;

		// relative motion based on last frame
		// used to estimate the initial pose of the current frame
		SE3 relative_motion_;

		int tracking_inliers_ = 0;

		//params
		int num_features_ = 200;
		int num_features_init_ = 100;
		int num_features_tracking_ = 50;
		int num_features_tracking_bad_ = 20;
		// does it mean when at least 80 features are needed for a frame to be called a key frame
		int num_features_needed_for_keyframe_ = 80;

		// feature detector, not ORB this time
		cv::Ptr<cv::GFTTDetector> gftt_;

	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		typedef std::shared_ptr<Frontend> Ptr;

		Frontend();

		// add a frame and calculate its localization results
		bool AddFrame(Frame::Ptr frame);

		void SetMap(Map::Ptr map) { map_ = map; }

		void SetBackend(std::shared_ptr<Backend> backend) { backend_ = backend; }

		void SetViewer(std::shared_ptr<Viewer> viewer) { viewer_ = viewer; }

		FrontendStatus GetStatus() const { return status_; }

		void SetCameras(Camera::Ptr left, Camera::Ptr right) {
			camera_left_ = left;
			camera_right_ = right;
		}
	};
}
#endif // !MYSLAM_FRONTEND_H
