// this is a sample of nonlinear optimization
// using either ceres and g2o can do min square method

// here we want to optimize the paramters in y = ax^2+bx+c

#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>

using namespace std;

// cost function defined in ceres
struct CURVE_FITTING_COST {

	// it means _x is the variable, _y is the target function
	CURVE_FITTING_COST(double x, double y) :
		_x(x), _y(y) {}

	// residual computation, ie, the error item
	template<typename T>
	bool operator() (const T* const abc, // model parameters, 3dim
		T* residual) const
	{
		residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]);
		return true;
	}

	const double _x, _y;
};

int main(int argc, char** argv) {
	// real parameters
	double ar = 1.0, br = 2.0, cr = 1.0;
	// estimation init
	double ae = 2.0, be = -1.0, ce = 5.0;

	int N = 100;
	double w_sigma = 1.0;
	double inv_sigma = 1.0 / w_sigma;
	cv::RNG rng;

	vector<double> x_data, y_data;
	for (int i = 0; i < N; ++i) {
		double x = i / 100.0;
		x_data.push_back(x);
		y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
	}

	double abc[3] = { ae, be, ce };

	// build a min square problem
	ceres::Problem problem;

	for (int i = 0; i < N; ++i) {
		problem.AddResidualBlock(
			new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(
				new CURVE_FITTING_COST(x_data[i], y_data[i])
				),
			nullptr,	// no kernel function here
			abc	// parameters to be estimated
		);
	}

	// set up the solver
	ceres::Solver::Options options;

	options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
	options.minimizer_progress_to_stdout = true;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	//cout << summary.BriefReport() << endl;
	cout << "estimated a,b,c = " << abc[0] << " " << abc[1] << " " << abc[2];
	cout << endl;

	return 0;
}