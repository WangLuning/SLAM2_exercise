#include <iostream>
using namespace std;

#include <Eigen/Core>
#include <Eigen/Dense>
using namespace Eigen;

#include <ctime>
int eigen_lib(int argc, char** argv) {
	cout << "hello world" << endl;

	// define a matrix
	Matrix<float, 2, 3> matrix_23;
	// define a vector to be double type
	Vector3d v_3d;
	// define a Matrix3d equivalent to Matrix<double, 3, 3>
	Matrix3d matrix_33 = Matrix3d::Zero();

	matrix_23 << 1, 2, 3, 4, 5, 6;
	cout << "matrix 2*3 from 1 to 6: \n" << matrix_23 << endl;

	// traverse the elements in the matrix
	cout << "print matrix 2*3: \n";
	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 3; ++j)
			cout << matrix_23(i, j) << '\t';
		cout << endl;
	}

	// need type conversion before matrix calculation
	// unlike cpp, Eigen will not do it automatically
	Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
	cout << "multiplication result: \n" << result << endl;

	// equation solver
	// matrix_NN * x = v_Nd
	Matrix<double, 50, 50> matrix_NN = MatrixXd::Random(50, 50);
	matrix_NN = matrix_NN * matrix_NN.transpose();
	Matrix<double, 50, 1> v_Nd = MatrixXd::Random(50, 1);

	// if using reverse matrix to calculate
	clock_t time_start = clock();
	Matrix<double, 50, 1> x = matrix_NN.inverse() * v_Nd;
	cout << "time of inverse is: " << 1000 * (clock() - time_start) / (double)CLOCKS_PER_SEC << "ms" << endl;

	// if using matrix decomposition to solve
	time_start = clock();
	x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
	cout << "time for Qr decomposition is: "<< 1000 * (clock() - time_start) / (double)CLOCKS_PER_SEC << "ms" << endl;

	return 0;
}
