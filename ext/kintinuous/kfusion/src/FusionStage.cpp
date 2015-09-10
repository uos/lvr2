#include <kfusion/FusionStage.hpp>

// default constructor
FusionStage::FusionStage(MeshPtr mesh, string mesh_name) : AbstractStage()
	, mesh_(mesh), mesh_count_(0), mesh_name_(mesh_name)
{

}

void FusionStage::firstStep() { /* skip */ };

void FusionStage::step()
{
	auto mesh_work = boost::any_cast<pair<MeshPtr, bool> >(getInQueue()->Take());
	bool last_shift = mesh_work.second;
	MeshPtr opti_mesh = mesh_work.first;
	string mesh_notice = ("#### D:                Mesh Fusion " +  to_string(mesh_count_) + "    ####");
	ScopeTime* fusion_time = new ScopeTime(mesh_notice.c_str());
	if(mesh_count_ == 0)
		mesh_ = opti_mesh;
	else
		mesh_->addMesh(opti_mesh);
	getOutQueue()->Add(mesh_);
	mesh_count_++;
	delete fusion_time;
	if(last_shift)
		done(true);
}
void FusionStage::lastStep()
{ 
	// plane_iterations, normal_threshold, min_plan_size, small_region_threshold
	//meshPtr_->optimizePlanes(3, 0.85, 7, 10, true);
	//meshPtr_->fillHoles(30);
	//meshPtr_->optimizePlaneIntersections();
	//min_plan_size
	//meshPtr_->restorePlanes(7);
	//meshPtr_->finalizeAndRetesselate(false, 0.01);
	//transformMeshBack();
	std::cout << "Global amount of vertices: " << mesh_->meshSize() << endl;
	std::cout << "Global amount of faces: " << mesh_->getFaces().size() << endl;
	mesh_->finalize();
	ModelPtr m( new Model( mesh_->meshBuffer() ) );
	ModelFactory::saveModel( m, mesh_name_);
	//ModelFactory::saveModel( m, "./test_mesh.ply"); 
}
