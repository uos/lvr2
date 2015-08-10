#include <kfusion/OptimizeStage.hpp>

// default constructor
OptimizeStage::OptimizeStage() : AbstractStage()
{
	mesh_count_ = 0;
}

void OptimizeStage::firstStep() { /* skip */ };

void OptimizeStage::step()
{
	auto mesh_work = boost::any_cast<pair<MeshPtr, bool> >(getInQueue()->Take());
	bool last_shift = mesh_work.second;
	MeshPtr act_mesh = mesh_work.first;
	act_mesh->optimizePlanes(3, 0.83, 7, 40);
	//act_mesh->optimizePlaneIntersections();
	MeshPtr tmp_pointer = act_mesh->retesselateInHalfEdge();
	std::cout << "            ####     3 Finished optimisation number: " << mesh_count_ << "   ####" << std::endl;
	mesh_count_++;
	getOutQueue()->Add(pair<MeshPtr, bool>(tmp_pointer, last_shift));
	delete act_mesh;
	if(last_shift)
		done(true);
}
void OptimizeStage::lastStep()	{ /* skip */ };
