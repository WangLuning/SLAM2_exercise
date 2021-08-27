// unidistort an image from metrics of the camera
#include<opencv2/opencv.hpp>
#include<string>
using namespace std;
string image_file = "C:\\Users\\Administrator\\Desktop\\eigen_test\\images\\dog.jpg";

int undistort_image(int argc, char** argv) {
	// distorted parameters
	double k1 = -0.28340811, k2 = 0.07395907,\
		p1 = 0.00019359, p2 = 1.76187114e-05;

	// internal parameters
	double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;

	cv::Mat image = cv::imread(image_file, 0);
	int rows = image.rows, cols = image.cols;
	cv::Mat image_undistort = cv::Mat(rows, cols, CV_8UC1);

	// calculate undistorted image
	for (int v = 0; v < rows; ++v) {
		for (int u = 0; u < cols; ++u) {
			// u,v is called the pixel coordinate
			// x, y is the physical projection panel
			// there is a real object pane
			// the relationship is, real object goes through a hole to project on projectional panel, and then zoomed to pixel coordinate
			double x = (u - cx) / fx, y = (v - cy) / fy;
			double r = sqrt(x * x + y * y);
			double x_undistorted = x * (1 + k1 * r * r + k2 * r * r * r * r) + 2 * p1 * x * y + p2 * (r * r + 2 * x * x);
			double y_undistorted = y * (1 + k1 * r * r + k2 * r * r * r * r) + p1 * (r * r + 2 * y * y) + 2 * p2 * x * y;
			double u_undistorted = fx * x_undistorted + cx, \
				v_undistorted = fy * y_undistorted + cy;

			// interpolation
			if (u_undistorted >= 0 && v_undistorted >= 0 && u_undistorted < cols && v_undistorted < rows) {
				image_undistort.at<uchar>(v, u) = image.at<uchar>((int)v_undistorted, (int)u_undistorted);
			}
			else {
				image_undistort.at<uchar>(v, u) = 0;
			}
		}
	}

	// show the undistorted image
	cv::imshow("distorted", image);
	cv::imshow("undistorted", image_undistort);
	cv::waitKey();
	return 0;
}