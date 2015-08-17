#include <kfusion/OptimizeStage.hpp>

// default constructor
OptimizeStage::OptimizeStage() : AbstractStage()
{
	mesh_count_ = 0;
}

void OptimizeStage::firstStep() { /* skip */ };

void OptimizeStage::step()
{
	auto mesh_work = boost::any_cast<pair<pair<MeshPtr, bool>, vector<kfusion::ImgPose*> > >(getInQueue()->Take());
	bool last_shift = mesh_work.first.second;
	MeshPtr act_mesh = mesh_work.first.first;
	act_mesh->optimizePlanes(3, 0.83, 7, 40);
	//act_mesh->optimizePlaneIntersections();
	MeshPtr tmp_pointer = act_mesh->retesselateInHalfEdge();
	std::cout << "            ####     3 Finished optimisation number: " << mesh_count_ << "   ####" << std::endl;
	mesh_count_++;
	getOutQueue()->Add(pair<pair<MeshPtr, bool>, vector<kfusion::ImgPose*> >(
				pair<MeshPtr, bool>(act_mesh, last_shift), mesh_work.second));
	delete act_mesh;
	if(last_shift)
		done(true);
}
void OptimizeStage::lastStep()	{ /* skip */ };
