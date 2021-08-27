#pragma once

#ifndef MYSLAM_MAPPOINT_H
#define MYSLAM_MAPPOINT_H

#include "common_include.h"

namespace myslam {
	struct Frame;
	struct Feature;

	// this is a class for mappoints
	// formed after triangulation, added as landmarks
	struct MapPoint {
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		typedef std::shared_ptr<MapPoint> Ptr;

		unsigned long id_ = 0;
		bool is_outlier_ = false;
		Vec3 pos_ = Vec3::Zero();
		std::mutex data_mutex_;
		// observed by feature matching
		int observed_times_ = 0;
		std::list<std::weak_ptr<Feature>> observations_;

		MapPoint() {}
		MapPoint(long id, Vec3 position);

		Vec3 Pos() {
			std::unique_lock<std::mutex> lck(data_mutex_);
			return pos_;
		}

		void SetPos(const Vec3& pos) {
			std::unique_lock<std::mutex> lck(data_mutex_);
			pos_ = pos;
		}

		void AddObservation(std::shared_ptr<Feature> feature) {
			std::unique_lock<std::mutex> lck(data_mutex_);
			observations_.push_back(feature);
			observed_times_++;
		}

		static MapPoint::Ptr CreateNewMappoint();
	};
}
#endif // !MYSLAM_MAPPOINT_H
