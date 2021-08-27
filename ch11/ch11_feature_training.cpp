// here we want to loop detection
// we need to use feature points (descriptors) to tell how similar the two pictures are
// so we extract feature and store in BoW
// the BoW vocab should be large enough to include all possible features for a scenario
// so we need to collect new vocab when the camera is running in a new scenario

#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	// read the 10 images to decide which two of them are in the loop
	vector<Mat> images;
	for (int i = 0; i < 10; ++i) {
		images.push_back(imread("../images/build_dict/" + to_string(i + 1) + ".png"));
	}

	cout << images.size() << endl;

	// detect ORB features
	Ptr<Feature2D> detector = ORB::create();
	vector<Mat> descriptors;
	for (Mat& image : images) {
		vector<KeyPoint> keypoints;
		Mat descriptor;
		detector->detectAndCompute(image, Mat(), keypoints, descriptor);
		descriptors.push_back(descriptor);
	}

	// using the above descriptor to build the BoW
	cout << "creating vocab..." << endl;
	DBoW3::Vocabulary vocab;
	// do k-means clustering based on the descriptors given
	vocab.create(descriptors);
	cout << "vocab info: " << vocab << endl;
	vocab.save("vocab10.yml.gz");

	return 0;
}