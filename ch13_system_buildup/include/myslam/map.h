#pragma once
#ifndef MYSLAM_MAP_H
#define MYSLAM_MAP_H

#include "common_include.h"
#include "frame.h"
#include "mappoint.h"

namespace myslam {
	// frontend calls InsertKeyframe and InsertMapPoint to add new frame and new mappoint
	// backend maintains the map
	class Map {
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		typedef std::shared_ptr<Map> Ptr;
		typedef std::unordered_map<unsigned long, MapPoint::Ptr> LandmarksType;
		typedef std::unordered_map<unsigned long, Frame::Ptr> KeyframesType;

		Map() {}

		// add a new key frame
		void InsertKeyFrame(Frame::Ptr frame);

		// add a new mappoint
		void InsertMapPoint(MapPoint::Ptr map_point);

		LandmarksType GetAllMapPoints() {
			std::unique_lock<std::mutex> lck(data_mutex_);
			return landmarks_;
		}

		// get all active mappoints
		LandmarksType GetActiveMapPoints() {
			std::unique_lock<std::mutex> lck(data_mutex_);
			return active_landmarks_;
		}

		// obtain active keyframes
		KeyframesType GetActiveKeyFrames() {
			std::unique_lock<std::mutex> lck(data_mutex_);
			return active_keyframes_;
		}

		// clear map
		void CleanMap();

	private:
		// set old keyframes as inactive
		void RemoveOldKeyframe();

		std::mutex data_mutex_;

		// all landmarks
		LandmarksType landmarks_;
		LandmarksType active_landmarks_;
		KeyframesType keyframes_;
		KeyframesType active_keyframes_;

		Frame::Ptr current_frame_ = nullptr;
		
		// does it mean we allow at most 7 active frames at a time?
		int num_active_keyframes_ = 7;
	};
}
#endif // !MYSLAM_MAP_H
