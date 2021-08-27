#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void find_feature_matches(
	const Mat& img_1, const Mat& img_2,
	std::vector<KeyPoint>& keypoints_1, std::vector<KeyPoint>& keypoints_2,
	std::vector<DMatch>& matches
);

void pose_estimation_2d2d(
	const std::vector<KeyPoint> &keypoints_1,
	const std::vector<KeyPoint> &keypoints_2,
	const std::vector<DMatch> &matches,
	Mat &R, Mat &t
);

void triangulation(
	const vector<KeyPoint> &keypoints_1,
	const vector<KeyPoint> &keypoints_2,
	const std::vector<DMatch> &matches,
	const Mat &R, const Mat &t,
	vector<Point3d> &points
);

inline cv::Scalar get_color(float depth) {
	float up_th = 50, low_th = 10, th_range = up_th - low_th;
	if (depth > up_th) {
		depth = up_th;
	}
	if (depth < low_th) {
		depth = low_th;
	}
	return cv::Scalar(255 * depth / th_range, 0, 255 * (1 - depth) / th_range);

}

Point2f pixel2cam(const Point2d& p, const Mat& K);

int main(int argc, char** argv) {
	// load two images from 2 2D camera
	Mat img_1 = imread("..\\images\\move1.png", IMREAD_COLOR);
	Mat img_2 = imread("..\\images\\move2.png", IMREAD_COLOR);

	// ***********************************************
	// find matching feature points
	// this is the beginning of all following steps
	vector<KeyPoint> keypoints_1, keypoints_2;
	vector<DMatch> matches;
	find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
	cout << "total matches found: " << matches.size() << endl;

	// ***********************************************
	//estimate the movement between two pictures
	// find rotation and translation parameters
	Mat R, t;
	pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

	// ***********************************************
	//triangulation
	vector<Point3d> points;
	triangulation(keypoints_1, keypoints_2, matches, R, t, points);

	// ***********************************************
	// verify triangulation and projection of eigen points
	Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
	Mat img1_plot = img_1.clone();
	Mat img2_plot = img_2.clone();
	for (int i = 0; i < matches.size(); ++i) {
		// first img
		float depth1 = points[i].z;
		cout << "depth: " << depth1 << endl;
		Point2d pt1_cam = pixel2cam(keypoints_1[matches[i].queryIdx].pt, K);
		cv::circle(img1_plot, keypoints_1[matches[i].queryIdx].pt, 2, get_color(depth1), 2);

		// second img
		// use R, t to convert "points" to the coordinate of the second camera
		Mat pt2_trans = R * (Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
		float depth2 = pt2_trans.at<double>(2, 0);
		cv::circle(img2_plot, keypoints_2[matches[i].trainIdx].pt, 2, get_color(depth2), 2);
	}

	// ***********************************************
	cv::imshow("img_1", img1_plot);
	cv::imshow("img_2", img2_plot);
	cv::waitKey();

	return 0;
}

void find_feature_matches(const Mat& img_1, const Mat& img_2,
	std::vector<KeyPoint>& keypoints_1, std::vector<KeyPoint>& keypoints_2,
	std::vector<DMatch>& matches) {
	// ORB the same way as before
	// always do feature matching for two images first
	Mat descriptors_1, descriptors_2;
	Ptr<FeatureDetector> detector = ORB::create();
	Ptr<DescriptorExtractor> descriptor = ORB::create();
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

	// FAST detection
	detector->detect(img_1, keypoints_1);
	detector->detect(img_2, keypoints_2);

	// calculate BRIEF descriptors
	descriptor->compute(img_1, keypoints_1, descriptors_1);
	descriptor->compute(img_2, keypoints_2, descriptors_2);

	// matching the BRIEF descriptors using Hamming distance
	vector<DMatch> match;
	matcher->match(descriptors_1, descriptors_2, match);

	// filter the matched points
	double min_dist = 10000, max_dist = 0;
	for (int i = 0; i < descriptors_1.rows; ++i) {
		double dist = match[i].distance;
		min_dist = (dist < min_dist) ? dist : min_dist;
		max_dist = (dist > max_dist) ? dist : max_dist;
	}

	printf("-- max dist: %f\n", max_dist);
	printf("-- min dist: %f\n", min_dist);

	for (int i = 0; i < descriptors_1.rows; ++i) {
		if (match[i].distance <= max(2 * min_dist, 30.0)) {
			matches.push_back(match[i]);
		}
	}
}

// this is what epipolar constraint is doing
void pose_estimation_2d2d(
	const std::vector<KeyPoint>& keypoints_1,
	const std::vector<KeyPoint>& keypoints_2,
	const std::vector<DMatch>& matches,
	Mat& R, Mat& t) {
	// camera intrinsic parameters
	Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

	// convert matching points to Point2f format
	vector<Point2f> points1, points2;

	for (int i = 0; i < (int)matches.size(); ++i) {
		points1.push_back(keypoints_1[matches[i].queryIdx].pt);
		points2.push_back(keypoints_2[matches[i].trainIdx].pt);
	}

	// compute essential matrix
	Point2d principal_point(325.1, 249.7);
	int focal_length = 521;
	Mat essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);

	// from essential matrix, we can recover the R and t
	recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
}

Point2f pixel2cam(const Point2d& p, const Mat& K) {
	return Point2f(
		(p.x - K.at<double>(0,2)) / K.at<double>(0,0),
		(p.y - K.at<double>(1,2)) / K.at<double>(1,1)
	);
}

// find the matching points that satisfy triangulation
// this yield depth, as an info in 3d points
void triangulation(
	const vector<KeyPoint>& keypoint_1,
	const vector<KeyPoint>& keypoint_2,
	const std::vector<DMatch>& matches,
	const Mat& R, const Mat& t,
	vector<Point3d>& points) {
	Mat T1 = (Mat_<float>(3, 4) <<
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0);
	Mat T2 = (Mat_<float>(3, 4) << 
		R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
		R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
		R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
		);

	Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
	vector<Point2f> pts_1, pts_2;
	for (DMatch m : matches) {
		// convert pixel to cam
		pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
		pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K));
	}

	Mat pts_4d;
	// how to call this function: 
	// cv::triangulatePoints(cam0, cam1, cam0pnts, cam1pnts, pnts3D);
	// The method used is Least Squares
	cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

	// convert to coordinate
	for (int i = 0; i < pts_4d.cols; ++i) {
		Mat x = pts_4d.col(i);
		// normalization
		x /= x.at<float>(3, 0);
		Point3d p(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0));
		points.push_back(p);
	}
}