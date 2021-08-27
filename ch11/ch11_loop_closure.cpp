#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

// compute similarity of the dict
int main(int argc, char** argv) {
	DBoW3::Vocabulary vocab("./vocab10.yml.gz");

	if (vocab.empty()) {
		cerr << "vocab does not exist" << endl;
		return 1;
	}

	vector<Mat> images;
	for (int i = 0; i < 10; ++i) {
		images.push_back(imread("../images/build_dict/" + to_string(i + 1) + ".png"));
	}

	// detect ORB
	Ptr<Feature2D> detector = ORB::create();
	vector<Mat> descriptors;
	for (Mat& image : images) {
		vector<KeyPoint> keypoints;
		Mat descriptor;
		detector->detectAndCompute(image, Mat(), keypoints, descriptor);
		descriptors.push_back(descriptor);
	}

	// either compare between every two images
	// or compare each image with the database

	// 1. compare between the images
	cout << "comparing images with images" << endl;
	for (int i = 0; i < images.size(); ++i) {
		DBoW3::BowVector v1;
		vocab.transform(descriptors[i], v1);
		for (int j = i; j < images.size(); ++j) {
			DBoW3::BowVector v2;
			vocab.transform(descriptors[j], v2);
			double score = vocab.score(v1, v2);
			cout << "image " << i << " vs image " << j << " : " << score << endl;
		}
		cout << endl;
	}

	// 2. compare to the database
	cout << "comparing the image to the database" << endl;
	DBoW3::Database db(vocab, false, 0);
	for (int i = 0; i < descriptors.size(); ++i) {
		db.add(descriptors[i]);
	}
	cout << "database info: " << db << endl;
	for (int i = 0; i < descriptors.size(); ++i) {
		DBoW3::QueryResults ret;
		db.query(descriptors[i], ret, 4);
		cout << "searching for image: " << i << " returns " << ret << endl;
	}
}