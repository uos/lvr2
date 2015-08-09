#include "FusionStage.hpp"

// default constructor
FusionStage::FusionStage(MeshPtr mesh) : AbstractStage(), mesh_(mesh), mesh_count_(0)
{

}

void FusionStage::firstStep() { /* skip */ };

void FusionStage::Step()
{
	auto mesh_work = boost::any_cast<pair<MeshPtr, bool> >(getInQueue()->Take());
	bool last_shift = mesh_work.second;
	MeshPtr opti_mesh = mesh_work.first;
	mesh_->addMesh(opti_mesh, opti_mesh->m_slice_verts);
	std::cout << "                        ####    4 Finished slice number: " << slice_count_ << "   ####" << std::endl;
	mesh_count_++;
	getOutQueue()->Add(pair<MeshPtr, bool>(tmp_pointer, grid_work.second));
	if(last_shift && getInQueue()->size() == 0)
		done(true);
}
void FusionStage::LastStep()
{ 
	// plane_iterations, normal_threshold, min_plan_size, small_region_threshold
	//meshPtr_->optimizePlanes(3, 0.85, 7, 10, true);
	//meshPtr_->fillHoles(30);
	//meshPtr_->optimizePlaneIntersections();
	//min_plan_size
	//meshPtr_->restorePlanes(7);
	//meshPtr_->finalizeAndRetesselate(false, 0.01);
	transformMeshBack();
	meshPtr_->finalize();
	ModelPtr m( new Model( meshPtr_->meshBuffer() ) );
	ModelFactory::saveModel( m, "./mesh_" + to_string(slice_count_) + ".ply");
	//ModelFactory::saveModel( m, "./test_mesh.ply"); 
}
