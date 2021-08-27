#include<iostream>
#include<chrono>
using namespace std;

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

int BasicImage(int argc, char** argv) {
	cv::Mat image;
	image = cv::imread("../images/dog.jpg");

	if (image.data == nullptr) {
		cerr << "file not found\n";
		return 0;
	}
	else
		cout << "dog file read\n";

	// basic properties of image
	cout << "width: " << image.cols << ", height: " << image.rows\
		<< ", number of channels: " << image.channels() << endl;
	cv::imshow("image", image);
	cv::waitKey(0);

	// check image type
	if (image.type() != CV_8UC1 && image.type() != CV_8UC3) {
		cout << "image type not supported" << endl;
		return 0;
	}

	// traverse the image
	chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
	for (size_t y = 0; y < image.rows; y++) {
		unsigned char* row_ptr = image.ptr<unsigned char>(y);
		for (size_t x = 0; x < image.cols; x++) {
			unsigned char* data_ptr = &row_ptr[x * image.channels()];

			for (int c = 0; c != image.channels(); ++c) {
				unsigned char data = data_ptr[c];
			}
		}
	}

	chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
	chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
	cout << "traverse duration: " << time_used.count() << " s" << endl;

	// this is not a deep copy, but a reference
	cv::Mat image_another = image;
	image_another(cv::Rect(0, 0, 100, 100)).setTo(0);
	cv::imshow("image: ", image);
	cv::waitKey(0);

	// use clone to deep copy
	cv::Mat image_clone = image.clone();
	image_clone(cv::Rect(0, 0, 100, 100)).setTo(255);
	cv::imshow("image: ", image);
	cv::imshow("image cloned: ", image_clone);
	cv::waitKey(0);

	cv::destroyAllWindows();
	return 0;
}
