#include <kfusion/OptimizeStage.hpp>

// default constructor
OptimizeStage::OptimizeStage(Options* options) : AbstractStage()
	,mesh_count_(0), options_(options), bounding_counter(0),texture_counter(0),pic_count_(0)
{
	 optiMesh_ = NULL;
	timestamp.setQuiet(!options_->verbose());
}

void OptimizeStage::firstStep() { }

void OptimizeStage::step()
{
	auto mesh_work = boost::any_cast<pair<pair<MeshPtr, bool>, vector<kfusion::ImgPose*> > >(getInQueue()->Take());
	bool last_shift = mesh_work.first.second;
	MeshPtr act_mesh = mesh_work.first.first;
	string mesh_notice = ("#### C:            Mesh Optimization " +  to_string(mesh_count_) + "    ####");
	ScopeTime* opti_time = new ScopeTime(mesh_notice.c_str());
	if(optiMesh_ == NULL)
		optiMesh_ = act_mesh;
	else
		optiMesh_->addMesh(act_mesh);
    std::vector<kfusion::ImgPose*> image_poses_buffer = mesh_work.second;
	std::cout << "Loaded " << image_poses_buffer.size() << " Images. " << std::endl;
	// Set recursion depth for region growing
	if(options_->getDepth())
	{
		optiMesh_->setDepth(options_->getDepth());
	}
	if(options_->getDanglingArtifacts())
	{
		optiMesh_->removeDanglingArtifacts(options_->getDanglingArtifacts());
	}
	optiMesh_->cleanContours(options_->getCleanContourIterations());
	optiMesh_->setClassifier(options_->getClassifier());
	optiMesh_->optimizePlanes(options_->getPlaneIterations(),
					options_->getNormalThreshold(),
					options_->getMinPlaneSize(),
					options_->getSmallRegionThreshold(), false);
	optiMesh_->fillHoles(options_->getFillHoles());
	optiMesh_->optimizePlaneIntersections();
	//optiMesh_->restorePlanes(options_->getMinPlaneSize());
	MeshPtr tmp_pointer = optiMesh_->retesselateInHalfEdge(options_->getLineFusionThreshold(), options_->textures(), texture_counter);
	if(tmp_pointer == NULL)
		return;
	delete opti_time;
	mesh_count_++;

    ///texturing
	if(options_->textures()){
		int counter=0;
		int i;
		for(i=0;i<image_poses_buffer.size();i++){
			counter = tmp_pointer->projectAndMapNewImage(*(image_poses_buffer[i]));
		}
		pic_count_+=i;
		
		
		
		texture_counter += tmp_pointer->textures.size();
		
		meshBufferPtr = tmp_pointer->meshBuffer();
				
	}
	image_poses_buffer.resize(0);
	getOutQueue()->Add(pair<pair<MeshPtr, bool>, vector<ImgPose*> >(
				pair<MeshPtr, bool>(tmp_pointer, last_shift), mesh_work.second));
	if(last_shift)
		done(true);
}
void OptimizeStage::lastStep()	
{  
	 delete optiMesh_;
}
