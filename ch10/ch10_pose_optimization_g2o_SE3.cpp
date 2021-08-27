// in ch9 there is BA problem solution
// it is time consuming with the growing of feature points size
// so there are two methods to fasten the computation
// 1. sliding window approach, only calculate the previous N slides
//	but we need to remove the expired first slide every time, making H matrix not symmetric
// 2. pose optimization
//	originally BA calculates both camera pose and landmarks
//	if we use feature points / direct method to fix the landmarks, and only optimize camera poses
//	we no longer optimize the landmarks, then the calculation is a lot easier

// here we read the sphere.g2o file which contains perturbation
// check if we can optimize to fix it

// sphere.g2o file uses translation and quaternion to record pose
// so the VERTEX_SE3 in the file format is:
// ID, t_x, t_y, t_z, q_x, q_y, q_z, q_w
// it is quaternion, not Lie algebra

#include <iostream>
#include <fstream>
#include <string>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

using namespace std;

int main(int argc, char** argv) {
	ifstream fin("..\\images\\sphere.g2o");
	if (!fin) {
		cout << "file does not exist!" << endl;
		return 1;
	}

	// set g2o
	typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType;
	typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
	auto solver = new g2o::OptimizationAlgorithmLevenberg(
		g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>())
	);
	g2o::SparseOptimizer optimizer;
	optimizer.setAlgorithm(solver);
	optimizer.setVerbose(true);

	int vertexCnt = 0, edgeCnt = 0;
	while (!fin.eof()) {
		string name;
		fin >> name;
		if (name == "VERTEX_SE3:QUAT") {
			// this is the SE3 vertex
			g2o::VertexSE3* v = new g2o::VertexSE3();
			int index = 0;
			fin >> index;
			v->setId(index);
			v->read(fin);
			optimizer.addVertex(v);
			vertexCnt++;
			if (index == 0)
				v->setFixed(true);
		}
		else if (name == "EDGE_SE3:QUAT") {
			// this is the edge connecting SE3-SE3
			g2o::EdgeSE3* e = new g2o::EdgeSE3();
			int idx1, idx2;
			fin >> idx1 >> idx2;
			e->setId(edgeCnt++);
			e->setVertex(0, optimizer.vertices()[idx1]);
			e->setVertex(1, optimizer.vertices()[idx2]);
			e->read(fin);
			optimizer.addEdge(e);
		}
		if (!fin.good())
			break;
	}

	cout << "read total " << vertexCnt << " vertices, " << edgeCnt << " edges" << endl;
	cout << "optimizing" << endl;
	optimizer.initializeOptimization();
	optimizer.optimize(30);

	cout << "saving optmization results" << endl;
	optimizer.save("result.g2o");
}