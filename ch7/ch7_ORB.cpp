// ORB (Oriented FAST and Rotated BRIEF)
// about FAST: keypoint detection:
// must be augmented with pyramid schemes for scale, FAST-9, circular radius of 9
// about BRIEF:
// FAST does not contain orientation so we need BRIEF for in-plane rotation
// keypoint pointing to intensity center
// decretize the angles to 2 * pi / 30 for a precomputed lookup table for BRIEF patterns

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

using namespace std;
using namespace cv;

string move1_file = "C:\\Users\\Administrator\\Desktop\\eigen_test\\images\\move1.png";
string move2_file = "C:\\Users\\Administrator\\Desktop\\eigen_test\\images\\move2.png";

int ORB(int argc, char** argv) {
	Mat img_1 = imread(move1_file, cv::IMREAD_COLOR);
	Mat img_2 = imread(move2_file, cv::IMREAD_COLOR);
	assert(img_1.data != nullptr && img_2.data != nullptr);

	// initialization
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;
	Ptr<FeatureDetector> detector = ORB::create();
	Ptr<DescriptorExtractor> descriptor = ORB::create();
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

	// step 1: detect Oriented FAST edge point position
	chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
	detector->detect(img_1, keypoints_1);
	detector->detect(img_2, keypoints_2);

	// step2: according to the corner detected, compute the BRIEF descriptors
	descriptor->compute(img_1, keypoints_1, descriptors_1);
	descriptor->compute(img_2, keypoints_2, descriptors_2);
	chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
	chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
	cout << "extract ORB cost: " << time_used.count() << " seconds" << endl;

	Mat outimg1;
	drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imshow("ORB features", outimg1);

	// step3: match the two BRIEF descriptors, using Hamming distance
	vector<DMatch> matches;
	t1 = chrono::steady_clock::now();
	matcher->match(descriptors_1, descriptors_2, matches);
	t2 = chrono::steady_clock::now();
	time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
	cout << "match ORB cost: " << time_used.count() << " seconds" << endl;

	// step 4: filter the matched point, compute min and max distance
	auto min_max = minmax_element(matches.begin(), matches.end(),\
		[](const DMatch& m1, const DMatch& m2) {return m1.distance < m2.distance; });

	double min_dist = min_max.first->distance;
	double max_dist = min_max.second->distance;

	printf("-- max dist : %f \n", max_dist);
	printf("-- min dist : %f \n", min_dist);

	// when cur_dist > 2 * min_dist, we think it is an error in matching
	// but min_dist can be real small, so we set magic number 30 as lower bound
	std::vector<DMatch> good_matches;
	for (int i = 0; i < descriptors_1.rows; ++i) {
		if (matches[i].distance <= max(2 * min_dist, 30.0)) {
			good_matches.push_back(matches[i]);
		}
	}

	// step 5: draw the matched results
	Mat img_match;
	Mat img_goodmatch;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
	imshow("all matches", img_match);
	imshow("good matches", img_goodmatch);
	waitKey(0);

	return 0;
}