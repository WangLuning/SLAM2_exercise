// Lie is different from Cartisen space
// SO(3) is special orthogonal, the orthogonal coordinate like rotation
// SE(3) is special Euclid, including both rotation and translation

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "sophus/se3.hpp"

using namespace std;
using namespace Eigen;

int main(int argc, char** argv) {
	// rotate 90 degrees around z axis
	Matrix3d R = AngleAxisd(M_PI / 2, Vector3d(0, 0, 1)).toRotationMatrix();
	Quaterniond q(R);
	Sophus::SO3d SO3_R(R);
	Sophus::SO3d SO3_q(q);
	// both are equivalent
	cout << "SO(3) from matrix: " << SO3_R.matrix() << endl;
	cout << "SO(3) from quaternion: " << SO3_q.matrix() << endl;

	// use log to find its Lie algebra
	Vector3d so3 = SO3_R.log();
	cout << "so3 = " << so3.transpose() << endl;
	// hat means vector to skew-symmetric matrix
	cout << "so3 hat = " << Sophus::SO3d::hat(so3) << endl;
	// vee means skew-symmetric metrics to vector
	cout << "so3 hat vee = " << Sophus::SO3d::vee(Sophus::SO3d::hat(so3)).transpose() << endl;

	// update
	Vector3d update_so3(1e-4, 0, 0);
	Sophus::SO3d SO3_updated = Sophus::SO3d::exp(update_so3) * SO3_R;
	cout << "SO3 updated = " << SO3_updated.matrix() << endl;

	// similar operation to SE3
	Vector3d t(1, 0, 0);
	// build SE3 from R,t
	Sophus::SE3d SE3_Rt(R, t);
	// build form q,t
	// we have known that R, q are the same thing in essential
	Sophus::SE3d SE3_qt(q, t);
	cout << "SE3 from R,t = " << SE3_Rt.matrix() << endl;
	cout << "SE3 from q,t = " << SE3_qt.matrix() << endl;

	// SE(3) is a 6dim vector
	typedef Eigen::Matrix<double, 6, 1> Vector6d;
	Vector6d se3 = SE3_Rt.log();
	cout << "se3 = " << se3.transpose() << endl;
	// in Sophus, translate is put first in SE(3), then rotation
	cout << "se3 hat = " << Sophus::SE3d::hat(se3) << endl;
	cout << "se3 hat vee = " << Sophus::SE3d::vee(Sophus::SE3d::hat(se3)).transpose() << endl;

	// show the update
	Vector6d updated_se3;
	updated_se3.setZero();
	updated_se3(0, 0) = 1e-4;
	Sophus::SE3d SE3_updated = Sophus::SE3d::exp(updated_se3) * SE3_Rt;
	cout << "SE3 updated = " << SE3_updated.matrix() << endl;
}