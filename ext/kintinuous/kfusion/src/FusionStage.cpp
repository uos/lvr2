#include <kfusion/FusionStage.hpp>

// default constructor
FusionStage::FusionStage(MeshPtr mesh, double camera_target_distance, double voxel_size) : AbstractStage()
	, mesh_(mesh), mesh_count_(0), camera_target_distance_(camera_target_distance), voxel_size_(voxel_size)
{

}

void FusionStage::firstStep() { /* skip */ };

void FusionStage::step()
{
	auto mesh_work = boost::any_cast<pair<MeshPtr, bool> >(getInQueue()->Take());
	bool last_shift = mesh_work.second;
	MeshPtr opti_mesh = mesh_work.first;
	if(mesh_count_ == 0)
		mesh_ = opti_mesh;
	else
		mesh_->addMesh(opti_mesh);
	std::cout << "                        ####    4 Finished slice number: " << mesh_count_ << "   ####" << std::endl;
	getOutQueue()->Add(mesh_);
	mesh_count_++;
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
	transformMeshBack();
	mesh_->finalize();
	ModelPtr m( new Model( mesh_->meshBuffer() ) );
	ModelFactory::saveModel( m, "./mesh_" + to_string(mesh_count_) + ".ply");
	//ModelFactory::saveModel( m, "./test_mesh.ply"); 
}

void FusionStage::transformMeshBack()
{
	for(auto vert : mesh_->getVertices())
	{
		// calc in voxel
		vert->m_position.x 	*= voxel_size_;				
		vert->m_position.y 	*= voxel_size_;				
		vert->m_position.z 	*= voxel_size_;			
		//offset for cube coord to center coord
		vert->m_position.x 	-= 1.5;				
		vert->m_position.y 	-= 1.5;				
		vert->m_position.z 	-= 1.5 - camera_target_distance_;				
		
		//offset for cube coord to center coord
		vert->m_position.x 	-= 150;				
		vert->m_position.y 	-= 150;				
		vert->m_position.z 	-= 150;
	}
}
