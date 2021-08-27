// using ORB can be slow since it needs descriptor
// we use optical flow to replace descriptors
// we still need detecting keypoints

// here we test and compare three methods
// 1. single level LK
// 2. multi level LK
// 3. CV library

// for LK the formula is
// I(x+dx, y+dy, t+dt) = I(x, y, t)

#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace cv;


class OpticalFlowTracker {
private:
	const Mat& img1;
	const Mat& img2;
	const vector<KeyPoint>& kp1;
	vector<KeyPoint>& kp2;
	vector<bool>& success;
	bool inverse = true;
	bool has_initial = false;

public:
	// this is the initializer
	OpticalFlowTracker(
		const Mat& img1_,
		const Mat& img2_,
		const vector<KeyPoint>& kp1_,
		vector<KeyPoint>& kp2_,
		vector<bool>& success_,
		bool inverse_ = true, bool has_initial_ = false
	) : img1(img1_), img2(img2_), kp1(kp1_), kp2(kp2_), success(success_), inverse(inverse_), has_initial(has_initial_)
	{}

	void calculateOpticalFlow(const Range& range);
};

/*
kp1 is from img1 has keypoints
kp2 is from img2, but if empty, use the guess from kp1
success is used to show if a keypoint is tracked successfully or not
inverse is whether using an inverse formula
*/
void OpticalFlowSingleLevel(
	const Mat& img1, const Mat& img2,
	const vector<KeyPoint>& kp1, vector<KeyPoint>& kp2,
	vector<bool>& success,
	bool inverse = false,
	bool has_initial_guess = false
);

/*
this is different from single level since we are using multiple levels
it is like zoom out a little bit
coz single level may be trapped in local optimal
so we need to zoom out to make the img ignore local details before zooming in
*/
void OpticalFlowMultiLevel(
	const Mat& img1,
	const Mat& img2,
	const vector<KeyPoint>& kp1,
	vector<KeyPoint>& kp2,
	vector<bool>& success,
	bool inverse = false
);

inline float GetPixelValue(const cv::Mat& img, float x, float y) {
	if (x < 0)
		x = 0;
	if (y < 0)
		y = 0;
	if (x >= img.cols)
		x = img.cols - 1;
	if (y >= img.rows)
		y = img.rows - 1;

	// find the nearest pixel to read
	uchar* data = &img.data[int(y) * img.step + int(x)];
	float xx = x - floor(x);
	float yy = y - floor(y);

	// estimate the pixel if x, y has float part
	return float(
		(1 - xx) * (1 - yy) * data[0] +
		xx * (1 - yy) * data[1] +
		(1 - xx) * yy * data[img.step] +
		xx * yy * data[img.step + 1]
		);
}

int main(int argc, char** argv) {
	Mat img1 = imread("..\\images\\LK1.png", 0);
	Mat img2 = imread("..\\images\\LK2.png", 0);

	vector<KeyPoint> kp1;
	// max num of keypoints to detect is 500
	Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20);
	detector->detect(img1, kp1);

	// track the detected keypoints in the second img
	// use single level LK to see if it works fine
	vector<KeyPoint> kp2_single;
	vector<bool> success_single;
	OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single);

	// use multi level LK to see if it works fine
	vector<KeyPoint> kp2_multi;
	vector<bool> success_multi;
	OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi, success_multi, true);

	// use opencv for validation
	vector<Point2f> pt1, pt2;
	for (auto& kp : kp1)
		pt1.push_back(kp.pt);
	vector<uchar> status;
	vector<float> error;
	cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error);

	// plot difference of above functions
	Mat img2_single = img2.clone();
	try
	{
		cv::cvtColor(img2, img2_single, COLOR_BGR2GRAY);
	}
	catch (cv::Exception& e)
	{
		cerr << e.msg << endl; // output exception message
	}
	for (int i = 0; i < kp2_single.size(); ++i) {
		if (success_single[i]) {
			cv::circle(img2_single, kp2_single[i].pt, 2, cv::Scalar(0, 250, 0), 2);
			cv::line(img2_single, kp1[i].pt, kp2_single[i].pt, cv::Scalar(0, 250, 0));
		}
	}

	Mat img2_multi = img2.clone();
	try
	{
		cv::cvtColor(img2, img2_multi, COLOR_BGR2GRAY);
	}
	catch (cv::Exception& e)
	{
		cerr << e.msg << endl; // output exception message
	}
	for (int i = 0; i < kp2_multi.size(); ++i) {
		if (success_multi[i]) {
			cv::circle(img2_multi, kp2_multi[i].pt, 2, cv::Scalar(0, 250, 0), 2);
			cv::line(img2_multi, kp1[i].pt, kp2_multi[i].pt, cv::Scalar(0, 250, 0));
		}
	}

	Mat img2_cv = img2.clone();
	try
	{
		cv::cvtColor(img2, img2_cv, COLOR_BGR2GRAY);
	}
	catch (cv::Exception& e)
	{
		cerr << e.msg << endl; // output exception message
	}
	for (int i = 0; i < pt2.size(); ++i) {
		if (status[i]) {
			cv::circle(img2_cv, pt2[i], 2, cv::Scalar(0, 250, 0), 2);
			cv::line(img2_cv, pt1[i], pt2[i], cv::Scalar(0, 250, 0));
		}
	}

	cv::imshow("tracked single level", img2_single);
	cv::imshow("tracked multi level", img2_multi);
	cv::imshow("tracked opencv", img2_cv);
	cv::waitKey(0);

	return 0;
}

void OpticalFlowSingleLevel(
	const Mat& img1, const Mat& img2,
	const vector<KeyPoint>& kp1,
	vector<KeyPoint>& kp2,
	vector<bool>& success,
	bool inverse, bool has_initial
) {
	kp2.resize(kp1.size());
	success.resize(kp1.size());
	OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, inverse, has_initial);
	parallel_for_(Range(0, kp1.size()), std::bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, placeholders::_1));
}

void OpticalFlowTracker::calculateOpticalFlow(const Range& range) {
	// parameters
	int half_path_size = 4;
	int iterations = 10;
	for (size_t i = range.start; i < range.end; ++i) {
		auto kp = kp1[i];
		double dx = 0, dy = 0;
		if (has_initial) {
			dx = kp2[i].pt.x - kp.pt.x;
			dy = kp2[i].pt.y - kp.pt.y;
		}

		double cost = 0, lastCost = 0;
		bool succ = true;

		// Gauss-Newton method
		Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
		Eigen::Vector2d b = Eigen::Vector2d::Zero();
		Eigen::Vector2d J;
		for (int iter = 0; iter < iterations; ++iter) {
			if (inverse == false) {
				H = Eigen::Matrix2d::Zero();
				b = Eigen::Vector2d::Zero();
			}
			else {
				b = Eigen::Vector2d::Zero();
			}

			cost = 0;

			for (int x = -half_path_size; x < half_path_size; ++x) {
				for (int y = -half_path_size; y < half_path_size; ++y) {
					double error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) -
						GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);
					if (!inverse) {
						J = -1.0 * Eigen::Vector2d(
							0.5 * (GetPixelValue(img2, kp.pt.x + dx + x + 1, kp.pt.y + dy + y) -
								GetPixelValue(img2, kp.pt.x + dx + x - 1, kp.pt.y + dy + y)),
							0.5 * (GetPixelValue(img2, kp.pt.x + dx + x + x, kp.pt.y + dy + y + 1) -
								GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y - 1))
						);
					}
					else if (iter == 0) {
						// J does not change when dx, dy is updated
						J = -1.0 * Eigen::Vector2d(
							0.5 * (GetPixelValue(img1, kp.pt.x + x + 1, kp.pt.y + y) -
								GetPixelValue(img1, kp.pt.x + x - 1, kp.pt.y + y)),
							0.5 * (GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y + 1) -
								GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y - 1))
						);
					}
					// compute H, b and cost
					b += -error * J;
					cost += error * error;
					if (!inverse || iter == 0) {
						H += J* J.transpose();
					}
				}
			}

			Eigen::Vector2d update = H.ldlt().solve(b);

			if (std::isnan(update[0])) {
				cout << "update is nan" << endl;
				succ = false;
				break;
			}
			if (iter > 0 && cost > lastCost) {
				break;
			}

			// update dx, dy
			dx += update[0];
			dy += update[1];
			lastCost = cost;
			succ = true;
			if (update.norm() < 1e-2)
				break;
		}

		success[i] = succ;
		kp2[i].pt = kp.pt + Point2f(dx, dy);
	}
}

void OpticalFlowMultiLevel(
	const Mat& img1, const Mat& img2,
	const vector<KeyPoint>& kp1,
	vector<KeyPoint>& kp2,
	vector<bool>& success,
	bool inverse
) {
	// build a pyramid, then for each layer use the same single level method
	int pyramids = 4;
	double pyramid_scale = 0.5;
	double scales[] = { 1.0, 0.5, 0.25, 0.125 };

	vector<Mat> pyr1, pyr2;
	for (int i = 0; i < pyramids; ++i) {
		if (i == 0) {
			pyr1.push_back(img1);
			pyr2.push_back(img2);
		}
		else {
			Mat img1_pyr, img2_pyr;
			cv::resize(pyr1[i - 1], img1_pyr, cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
			cv::resize(pyr2[i - 1], img2_pyr, cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
			pyr1.push_back(img1_pyr);
			pyr2.push_back(img2_pyr);
		}
	}

	// from coarse to fine bcoz we want to neglect details first
	vector<KeyPoint> kp1_pyr, kp2_pyr;
	for (auto& kp : kp1) {
		auto kp_top = kp;
		kp_top.pt *= scales[pyramids - 1];
		kp1_pyr.push_back(kp_top);
		kp2_pyr.push_back(kp_top);
	}

	for (int level = pyramids - 1; level >= 0; --level) {
		success.clear();
		OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_pyr, kp2_pyr, success, true);
		
		if (level > 0) {
			for (auto& kp : kp1_pyr)
				kp.pt /= pyramid_scale;
			for (auto& kp : kp2_pyr) {
				kp.pt /= pyramid_scale;
			}
		}
	}
}