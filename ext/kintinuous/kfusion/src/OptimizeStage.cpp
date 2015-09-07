#include <kfusion/OptimizeStage.hpp>

// default constructor
OptimizeStage::OptimizeStage(double camera_target_distance, double voxel_size, bool optimize) : AbstractStage()
	,mesh_count_(0), camera_target_distance_(camera_target_distance), voxel_size_(voxel_size), optimize_(optimize)
{
	timestamp.setQuiet(true);
}

void OptimizeStage::firstStep() { /* skip */ };

void OptimizeStage::step()
{
	auto mesh_work = boost::any_cast<pair<MeshPtr, bool> >(getInQueue()->Take());
	bool last_shift = mesh_work.second;
	MeshPtr act_mesh = mesh_work.first;
	if(optimize_)
	{
		act_mesh->optimizePlanes(3, 0.83, 7, 40);
		//act_mesh->optimizePlaneIntersections();
		act_mesh = act_mesh->retesselateInHalfEdge();
	}
	std::cout << "            ####     3 Finished optimisation number: " << mesh_count_ << "   ####" << std::endl;
	mesh_count_++;
	//transformMeshBack(tmp_pointer);
	transformMeshBack(act_mesh);
	//getOutQueue()->Add(pair<MeshPtr, bool>(tmp_pointer, last_shift));
	getOutQueue()->Add(pair<MeshPtr, bool>(act_mesh, last_shift));
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
