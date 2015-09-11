#include <kfusion/OptimizeStage.hpp>

// default constructor
OptimizeStage::OptimizeStage() : AbstractStage()
	,mesh_count_(0)
{
	timestamp.setQuiet(true);
}

void OptimizeStage::firstStep() { optiMesh_ = NULL; }

void OptimizeStage::step()
{
	auto mesh_work = boost::any_cast<pair<pair<MeshPtr, bool>, vector<kfusion::ImgPose*> > >(getInQueue()->Take());
	bool last_shift = mesh_work.first.second;
	MeshPtr act_mesh = mesh_work.first.first;
	string mesh_notice = ("#### C:            Mesh Optimization " +  to_string(mesh_count_) + "    ####");
	ScopeTime* opti_time = new ScopeTime(mesh_notice.c_str());
	ScopeTime* opo_time = new ScopeTime("All without retesselation");
	if(optiMesh_ == NULL)
		optiMesh_ = act_mesh;
	else
		optiMesh_->addMesh(act_mesh);
	timestamp.setQuiet(true);
	//optiMesh_->optimizePlanes(3, 0.8, 7, 0);
	optiMesh_->optimizePlanes(3, 0.8, 7, 10, true);
	optiMesh_->fillHoles(40);
	optiMesh_->optimizePlaneIntersections();
	delete opo_time;
	MeshPtr tmp_pointer = optiMesh_->retesselateInHalfEdge();
	delete opti_time;
	mesh_count_++;
	getOutQueue()->Add(pair<pair<MeshPtr, bool>, vector<ImgPose*> >(
				pair<MeshPtr, bool>(tmp_pointer, last_shift), mesh_work.second));
	if(last_shift)
		done(true);
}
void OptimizeStage::lastStep()	
{  
	 
}
