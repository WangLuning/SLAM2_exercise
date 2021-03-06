// how to estimante the movement of camera
// using 2D-2D feature matching

// epipolar constraint:
// use a set of 2D matching points to estimate the movement of camera
// it is called constraint because without knowing the real world position of point,
// from camera 1 we can constrain the matching point to be on the epipolar line in camera 2

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

void find_feature_matches(const Mat &img1_1,const Mat &img_2,
	std::vector<KeyPoint> & keypoints_1, std::vector<KeyPoint> &keypoints_2,
	std::vector<DMatch> &matches);

void pose_estimation_2d2d(
	std::vector<KeyPoint> keypoints_1, std::vector<KeyPoint> keypoints_2,
	std::vector<DMatch> matches,
	Mat &R, Mat &t);

// pixel coordinate to camera normalized coordinate
Point2d pixel2cam(const Point2d& p, const Mat& k);

int main(int argc, char** argv) {
	// load two images from 2 2D camera
	Mat img_1 = imread("..\\images\\move1.png", IMREAD_COLOR);
	Mat img_2 = imread("..\\images\\move2.png", IMREAD_COLOR);

	assert(img_1.data && img_2.data && "Cannot load images");

	vector<KeyPoint> keypoints_1, keypoints_2;
	vector<DMatch> matches;
	find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
	cout << "in total find : " << matches.size() << " matched points" << endl;

	// estimate the movement between the two images
	Mat R, t;
	pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

	// verifiy E = t^R*scale
	Mat t_x = (Mat_<double>(3,3) << 0, -t.at<double>(2,0), t.at<double>(1,0),
		t.at<double>(2,0), 0, -t.at<double>(0,0),
		-t.at<double>(1, 0), t.at<double>(0, 0), 0);

	cout << "t^R = " << endl << t_x * R << endl;

	// verify epipolar constraint
	Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
	for (DMatch m : matches) {
		Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
		Mat y1 = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
		Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
		Mat y2 = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
		Mat d = y2.t() * t_x * R * y1;
		cout << "epipolar contraint = " << d << endl;
	}
	return 0;
}

// this is the matching stereo vision we have written before
void find_feature_matches(const Mat& img_1, const Mat& img_2,
	std::vector<KeyPoint>& keypoints_1, std::vector<KeyPoint>& keypoints_2,
	std::vector<DMatch>& matches) {
	Mat descriptors_1, descriptors_2;
	Ptr<FeatureDetector> detector = ORB::create();
	Ptr<DescriptorExtractor> descriptor = ORB::create();
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

	// FAST
	detector->detect(img_1, keypoints_1);
	detector->detect(img_2, keypoints_2);

	// calcualte BRIEF descriptor
	descriptor->compute(img_1, keypoints_1, descriptors_1);
	descriptor->compute(img_2, keypoints_2, descriptors_2);

	// use Hamming to match the BRIEF descriptors
	vector<DMatch> match;
	matcher->match(descriptors_1, descriptors_2, match);

	// filter the matched points, not all are good quality
	double min_dist = 10000, max_dist = 0;
	for (int i = 0; i < descriptors_1.rows; ++i) {
		double dist = match[i].distance;
		min_dist = (dist < min_dist) ? dist : min_dist;
		max_dist = (dist > max_dist) ? dist : max_dist;
	}

	printf("-- max dist: %f\n", max_dist);
	printf("-- min_dist: %f\n", min_dist);

	// set lower bound of min_dist to be 30
	for (int i = 0; i < descriptors_1.rows; ++i) {
		if (match[i].distance <= max(2 * min_dist, 30.0)) {
			matches.push_back(match[i]);
		}
	}
}

Point2d pixel2cam(const Point2d& p, const Mat& K) {
	return Point2d(
		(p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
		(p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
	);
}

void pose_estimation_2d2d(std::vector<KeyPoint> keypoints_1,
	std::vector<KeyPoint> keypoints_2,
	std::vector<DMatch> matches,
	Mat& R, Mat& t) {
	// intrinsic of camera
	Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

	vector<Point2f> points1;
	vector<Point2f> points2;

	for (int i = 0; i < (int)matches.size(); ++i) {
		points1.push_back(keypoints_1[matches[i].queryIdx].pt);
		points2.push_back(keypoints_2[matches[i].trainIdx].pt);
	}

	// compute fundamental matrix
	Mat fundamental_matrix;
	fundamental_matrix = findFundamentalMat(points1, points2, 2);
	cout << "fundamental matrix is " << endl << fundamental_matrix << endl;

	// compute essential matrix
	Point2d principal_point(325.1, 249.7);
	double focal_length = 521;
	Mat essential_matrix;
	essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
	cout << "essential matrix is " << endl << essential_matrix << endl;

	// compute homography matrix
	Mat homography_matrix = findHomography(points1, points2, RANSAC, 3);
	cout << "homography_matrix is " << endl << homography_matrix << endl;

	//recover rotation and translation from essential matrix
	recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
	cout << "R is " << endl << R << endl;
	cout << "t is " << endl << t << endl;
}