#include <kfusion/OptimizeStage.hpp>

// default constructor
OptimizeStage::OptimizeStage(double camera_target_distance, double voxel_size) : AbstractStage()
	,mesh_count_(0), camera_target_distance_(camera_target_distance), voxel_size_(voxel_size)
{
}

void OptimizeStage::firstStep() { optiMesh_ = NULL; };

void OptimizeStage::step()
{
	auto mesh_work = boost::any_cast<pair<pair<MeshPtr, bool>, vector<kfusion::ImgPose*> > >(getInQueue()->Take());
	bool last_shift = mesh_work.first.second;
	MeshPtr act_mesh = mesh_work.first.first;
    	transformMeshBack(act_mesh);
	//if(optiMesh_ == NULL)
		optiMesh_ = act_mesh;
	//else
		//optiMesh_->addMesh(act_mesh);
	
	optiMesh_->optimizePlanes(3, 0.85, 7, 0);
	//act_mesh->optimizePlaneIntersections();
	MeshPtr tmp_pointer = optiMesh_->retesselateInHalfEdge();
	std::cout << "            ####     3 Finished optimisation number: " << mesh_count_ << "   ####" << std::endl;
	mesh_count_++;
	getOutQueue()->Add(pair<pair<MeshPtr, bool>, vector<kfusion::ImgPose*> >(
				pair<MeshPtr, bool>(tmp_pointer, last_shift), mesh_work.second));
	//delete act_mesh;
	if(last_shift)
		done(true);
}
void OptimizeStage::lastStep()	{ /* skip */ };

void OptimizeStage::transformMeshBack(MeshPtr mesh)
{
	for(auto vert : mesh->getVertices())
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
