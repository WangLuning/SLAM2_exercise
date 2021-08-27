#include <opencv2/opencv.hpp>
#include "../include/myslam/algorithm.h"
#include "../include/myslam/backend.h"
#include "../include/myslam/config.h"
#include "../include/myslam/feature.h"
#include "../include/myslam/frontend.h"
#include "../include/myslam/g2o_types.h"
#include "../include/myslam/map.h"
#include "../include/myslam/viewer.h"

namespace myslam {
	Frontend::Frontend() {
		gftt_ = cv::GFTTDetector::create(Config::Get<int>("num_features"), 0.01, 20);
		num_features_init_ = Config::Get<int>("num_features_init");
		num_features_ = Config::Get<int>("num_features");
	}

	bool Frontend::AddFrame(myslam::Frame::Ptr frame) {
		current_frame_ = frame;
		
		switch (status_) {
		case FrontendStatus::INITING:
			StereoInit();
			break;
		case FrontendStatus::TRACKING_GOOD:
		case FrontendStatus::TRACKING_BAD:
			Track();
			break;
		case FrontendStatus::LOST:
			Reset();
			break;
		}

		// after the above status switching,
		// change this processed frame to last frame
		last_frame_ = current_frame_;
		return true;
	}

	bool Frontend::Track() {
		if (last_frame_) {
			// this is only a init estimation based on previous pose
			current_frame_->SetPose(relative_motion_ * last_frame_->Pose());
		}

		// why do we need this?
		int num_track_last = TrackLastFrame();
		tracking_inliers_ = EstimateCurrentPose();

		if (tracking_inliers_ > num_features_tracking_) {
			// tracking good
			status_ = FrontendStatus::TRACKING_GOOD;
		}
		else if (tracking_inliers_ > num_features_tracking_bad_) {
			// tracking bad
			status_ = FrontendStatus::TRACKING_BAD;
		}
		else {
			status_ = FrontendStatus::LOST;
		}

		InsertKeyframe();
		relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse();

		if (viewer_)
			viewer_->addCurrentFrame(current_frame_);
		return true;
	}

	bool Frontend::InsertKeyframe() {
		// if we still have enough features to track, we do not need new frame
		if (tracking_inliers_ >= num_features_needed_for_keyframe_) {
			return false;
		}

		current_frame_->SetKeyFrame();
		map_->InsertKeyFrame(current_frame_);

		std::cout << "set frame" << current_frame_->id_ << " as a key frame" << current_frame_->keyframe_id_ << endl;

		// set the features in keyframe as new observation of map points
		SetObservationsForKeyFrame();
		// triangulate map points
		TriangulateNewPoints();
		// update backend on keyframes
		backend_->UpdateMap();

		if (viewer_)
			viewer_->UpdateMap();

		return true;
	}

	void Frontend::SetObservationsForKeyFrame() {
		for (auto& feat : current_frame_->features_left_) {
			auto mp = feat->map_point_.lock();

			// this is defined in mappoint.h
			// add this feature to unordered_map of feature points
			if (mp)
				mp->AddObservation(feat);
		}
	}

	int Frontend::TriangulateNewPoints() {
		std::vector<SE3> poses{ camera_left_->pose(), camera_right_->pose() };
		SE3 current_pose_Twc = current_frame_->Pose().inverse();
		int cnt_triangulated_pts = 0;

		for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
			if (current_frame_->features_left_[i]->map_point_.expired() &&
				current_frame_->features_right_[i] != nullptr) {
				std::vector<Vec3> points{
					camera_left_->pixel2camera(
						Vec2(current_frame_->features_left_[i]->position_.pt.x,
							current_frame_->features_left_[i]->position_.pt.y)),
					camera_right_->pixel2camera(
						Vec2(current_frame_->features_right_[i]->position_.pt.x,
							current_frame_->features_right_[i]->position_.pt.y))
				};
				Vec3 pworld = Vec3::Zero();

				if (triangulation(poses, points, pworld) && pworld[2] > 0) {
					auto new_map_point = MapPoint::CreateNewMappoint();
					pworld = current_pose_Twc * pworld;
					new_map_point->SetPos(pworld);
					new_map_point->AddObservation(current_frame_->features_left_[i]);
					new_map_point->AddObservation(current_frame_->features_right_[i]);

					current_frame_->features_left_[i]->map_point_ = new_map_point;
					current_frame_->features_right_[i]->map_point_ = new_map_point;
					map_->InsertMapPoint(new_map_point);
					cnt_triangulated_pts++;
				}
			}
		}

		std::cout << "new landmarks " << cnt_triangulated_pts << std::endl;
		return cnt_triangulated_pts;
	}

	// this is very similar to "Frontend::FindFeaturesInRight()"
	// both use LK method to save cost
	// here it is optical flow tracking between "last left" and "current left"
	int Frontend::TrackLastFrame() {
		// this tracking is between left and right images
		// use LK optical flow coz it is less expensive
		// we need to know the good tracking points between left and right
		
		std::vector<cv::Point2f> kps_last, kps_current;
		for (auto& kp : last_frame_->features_left_) {
			if (kp->map_point_.lock()) {
				auto mp = kp->map_point_.lock();
				// this is an estimated init value
				// it is not precise, that is why we use LK to optimize it
				auto px = camera_left_->world2pixel(mp->pos_, current_frame_->Pose());
				kps_last.push_back(kp->position_.pt);
				kps_current.push_back(cv::Point2f(px[0], px[1]));
			}
			else {
				kps_last.push_back(kp->position_.pt);
				kps_current.push_back(kp->position_.pt);
			}
		}

		std::vector<uchar> status;
		Mat error;
		cv::calcOpticalFlowPyrLK(last_frame_->left_img_, current_frame_->left_img_, kps_last, kps_current, status, error, cv::Size(11, 11), 3,
			cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
			cv::OPTFLOW_USE_INITIAL_FLOW);

		int num_good_pts = 0;

		for (size_t i = 0; i < status.size(); ++i) {
			if (status[i]) {
				cv::KeyPoint kp(kps_current[i], 7);
				Feature::Ptr feature(new Feature(current_frame_, kp));
				feature->map_point_ = last_frame_->features_left_[i]->map_point_;
				current_frame_->features_left_.push_back(feature);
				num_good_pts++;
			}
		}

		std::cout << "find " << num_good_pts << " in last image" << std::endl;
		return num_good_pts;
	}

	bool Frontend::StereoInit() {
		int num_features_left = DetectFeatures();
		int num_coor_features = FindFeaturesInRight();

		// features in right eye is too little to init the sys
		if (num_coor_features < num_features_init_) {
			return false;
		}

		bool build_map_success = BuildInitMap();
		if (build_map_success) {
			status_ = FrontendStatus::TRACKING_GOOD;
			if (viewer_) {
				viewer_->AddCurrentFrame(current_frame_);
				viewer_->UpdateMap();
			}
			return true;
		}
		return false;
	}

	int Frontend::DetectFeatures() {
		cv::Mat mask(current_frame_->left_img_.size(), CV_8UC1, 255);
		for (auto& feat : current_frame_->features_left_) {
			cv::rectangle(mask, feat->position_.pt - cv::Point2f(10, 10), feat->position_.pt + cv::Point2f(10, 10), 0, CV_FILLED);
		}

		std::vector<cv::KeyPoint> keypoints;
		gftt_->detect(current_frame_->left_img_, keypoints, mask);
		int cnt_detected = 0;
		for (auto& kp : keypoints) {
			current_frame_->features_left_.push_back(Feature::Ptr(new Feature(current_frame_, kp)));
			cnt_detected++;
		}

		std::cout << "Detected " << cnt_detected << " new features" << std::endl;
		return cnt_detected;
	}


	// this is very similar to "Frontend::TrackLastFrame()"
	// both use LK method to save cost
	// here it is optical flow tracking between "current left" and "current right"
	int Frontend::FindFeaturesInRight() {
		//use LK
		std::vector<cv::Point2f> kps_left, kps_right;
		for (auto& kp : current_frame_->features_left_) {
			kps_left.push_back(kp->position_.pt);
			auto mp = kp->map_point_.lock();
			if (mp) {
				// this is initial guess
				auto px = camera_right_->world2pixel(mp->pos_, current_frame_->Pose());
				kps_right.push_back(cv::Point2f(px[0], px[1]));
			}
			else {
				kps_right.push_back(kp->position_.pt);
			}
		}

		std::vector<uchar> status;
		Mat error;
		cv::calcOpticalFlowPyrLK(current_frame_->left_img_, current_frame_->right_img_, kps_left, kps_right, status, error, cv::Size(11, 11), 3,
			cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);

		int num_good_pts = 0;
		for (size_t i = 0; i < status.size(); ++i) {
			if (status[i]) {
				cv::KeyPoint kp(kps_right[i], 7);
				Feature::Ptr feat(new Feature(current_frame_, kp));
				feat->is_on_left_image_ = false;
				current_frame_->features_right_.push_back(feat);
				num_good_pts++;
			}
			else {
				current_frame_->features_right_.push_back(nullptr);
			}
		}

		std::cout << "find " << num_good_pts << " in the right image" << std::endl;
		return num_good_pts;
	}

	bool Frontend::BuildInitMap() {
		std::vector<SE3> poses(camera_left_->pose(), camera_right_->pose());
		size_t cnt_init_landmarks = 0;
		for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
			if (current_frame_->features_right_[i] == nullptr)
				continue;

			// create map from triangulation
			// we do not need to create map all the time
			std::vector<Vec3> points{
			camera_left_->pixel2camera(
				Vec2(current_frame_->features_left_[i]->position_.pt.x,
					current_frame_->features_left_[i]->position_.pt.y)),
			camera_right_->pixel2camera(
				Vec2(current_frame_->features_right_[i]->position_.pt.x,
					current_frame_->features_right_[i]->position_.pt.y))
			};
			Vec3 pworld = Vec3::Zero();

			if (triangulation(poses, points, pworld) && pworld[2] > 0) {
				auto new_map_point = MapPoint::CreateNewMappoint();
				new_map_point->SetPos(pworld);
				new_map_point->AddObservation(current_frame_->features_left_[i]);
				new_map_point->AddObservation(current_frame_->features_right_[i]);
				current_frame_->features_left_[i]->map_point_ = new_map_point;
				current_frame_->features_right_[i]->map_point_ = new_map_point;
				cnt_init_landmarks++;
				map_->InsertMapPoint(new_map_point);
			}
		}

		current_frame_->SetKeyFrame();
		map_->InsertKeyFrame(current_frame_);
		backend_->UpdateMap();

		std::cout << "init map with " << cnt_init_landmarks << " map points" << std::endl;
		return true;
	}

	bool Frontend::Reset() {
		return true;
	}
}