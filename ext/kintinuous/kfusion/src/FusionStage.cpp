#include <kfusion/FusionStage.hpp>

// default constructor
FusionStage::FusionStage(MeshPtr mesh, Options* options) : AbstractStage()
	, mesh_(mesh), mesh_count_(0), options_(options)
{

}

void FusionStage::firstStep() { /* skip */ };

void FusionStage::step()
{
	auto mesh_work = boost::any_cast<pair<pair<MeshPtr, bool>, vector<kfusion::ImgPose*> > >(getInQueue()->Take());
	bool last_shift = mesh_work.first.second;
	MeshPtr opti_mesh = mesh_work.first.first;
	string mesh_notice = ("#### D:                Mesh Fusion " +  to_string(mesh_count_) + "    ####");
	ScopeTime* fusion_time = new ScopeTime(mesh_notice.c_str());
	if(mesh_count_ == 0)
		mesh_ = opti_mesh;
	else
		mesh_->addMesh(opti_mesh, options_->textures());
	mesh_->fillHoles(options_->getFillHoles());
	//optiMesh_->restorePlanes(options_->getMinPlaneSize());
	getOutQueue()->Add(mesh_);
	mesh_count_++;
	delete fusion_time;
	if(last_shift)
		done(true);
}
void FusionStage::lastStep()
{
	std::cout << "Global amount of vertices: " << mesh_->meshSize() << endl;
	std::cout << "Global amount of faces: " << mesh_->getFaces().size() << endl;
	mesh_->finalize();
	ModelPtr m( new Model( mesh_->meshBuffer() ) );
	if(!options_->textures())
		ModelFactory::saveModel( m, string(options_->getOutput() + ".ply"));
	else
		ModelFactory::saveModel( m, string(options_->getOutput() + ".obj"));
}
