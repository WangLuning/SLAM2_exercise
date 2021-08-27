//关于双目相机如何拼左右眼两张图像
#include<opencv2/opencv.hpp>
#include<string>
#include <Eigen/Core>
#include <Eigen/Geometry>
//#include <pangolin/pangolin.h>

using namespace std;
string left_file = "C:\\Users\\Administrator\\Desktop\\eigen_test\\images\\left.png";
string right_file = "C:\\Users\\Administrator\\Desktop\\eigen_test\\images\\left.png";

void showPointCloud(
	const vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& pointcloud);

int stereo(int argc, char** argv) {
	// internal parameters
	double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
	// baseline: the distance between two cameras
	double b = 0.573;

	// read the image
	cv::Mat left = cv::imread(left_file, 0);
	cv::Mat right = cv::imread(right_file, 0);

	// magic numbers
	cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 96, 9, 8*9*9, 32*9*9, 1, 63, 10, 100, 32);
	cv::Mat disparity_sgbm, disparity;
	sgbm->compute(left, right, disparity_sgbm);
	disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);

	// generate the point cloud
	vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> pointcloud;

	for (int v = 0; v < left.rows; v += 2) {
		for (int u = 0; u < left.cols; u += 2) {
			if (disparity.at<float>(v, u) <= 10.0 || disparity.at<float>(v, u) >= 96.0)
				continue;

			// first three cols are x,y,z, the fourth col is color
			Eigen::Vector4d point(0, 0, 0, left.at<uchar>(v, u) / 255.0);

			// compute the position of point based on stereovision
			double x = (u - cx) / fx;
			double y = (v - cy) / fy;
			double depth = fx * b / (disparity.at<float>(v, u));
			point[0] = x * depth;
			point[1] = y * depth;
			point[2] = depth;

			pointcloud.push_back(point);

		}
	}

	cv::imshow("left file", left);
	cv::imshow("right file", right);
	cv::imshow("disparity", disparity / 96.0);
	cv::waitKey(0);
	// draw the pointcloud
	//showPointCloud(pointcloud);
	return 0;
}