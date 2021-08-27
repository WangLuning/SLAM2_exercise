// trajectory.txt file format:
// time, tx, ty, tz, qx, qy, qz, qw

#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <iostream>
#include <fstream>

using namespace std;
using namespace Eigen;

string trajectory_file = "../images/trajectory.txt";

void DrawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>>);

int draw(int argc, char** argv) {
	vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses;
	ifstream fin(trajectory_file);
	if (!fin) {
		std::cout << "cannot find trajectory input file" << endl;
		return 1;
	}

	// read the shift and quanternion from the file
	while (!fin.eof()) {
		double time, tx, ty, tz, qx, qy, qz, qw;
		fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
		Isometry3d Twr(Quaterniond(qw, qx, qy, qz));
		Twr.pretranslate(Eigen::Vector3d(tx, ty, tz));
		poses.push_back(Twr);
	}

	// draw trajectory in pangolin
	DrawTrajectory(poses);
	return 0;
}

void DrawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses) {
	// create pangolin window and plot the trajectory
	pangolin::CreateWindowAndBind("Trajectory Viewer", 8, 8);
}