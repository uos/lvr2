#include <kfusion/GridStage.hpp>

// default constructor
GridStage::GridStage(double voxel_size) : AbstractStage()
{
	grid_count_ = 0;
	voxel_size_  = voxel_size;
	bbox_ = BoundingBox<cVertex>(0.0, 0.0, 0.0, 300.0, 300.0, 300.0);
	bbox_.expand(300.0, 300.0, 300.0);
	timestamp.setQuiet(true);
}

void GridStage::firstStep() { /* omit */ };

void GridStage::step()
{
	auto slice_work = boost::any_cast<pair<TSDFSlice, bool> >(getInQueue()->Take());
	cv::Mat& tsdf_values = slice_work.first.tsdf_values_;
	Vec3i offset = slice_work.first.offset_;
	bool last_shift = slice_work.second;
	ScopeTime* grid_time = new ScopeTime("Grid Creation");
	Point* tsdf_ptr = tsdf_values.ptr<Point>();				
	TGrid* act_grid = NULL;
	if(last_grid_queue_.size() == 0)
		act_grid = new TGrid(voxel_size_, bbox_, tsdf_ptr, tsdf_values.cols, offset[0], offset[1], offset[2], NULL, true);
	else
		act_grid = new TGrid(voxel_size_, bbox_, tsdf_ptr, tsdf_values.cols, offset[0], offset[1], offset[2], last_grid_queue_.front(), true);
	std::cout << "    ####     1 Finished grid number: " << grid_count_ << "   ####" << std::endl;
	//grid_ptr->saveGrid("./slices/grid" + std::to_string(slice_count_) + ".grid");
	double recon_factor = (grid_time->getTime()/tsdf_values.cols) * 1000;
	//timeStats_.push_back(recon_factor);
	delete grid_time;
	if(last_grid_queue_.size() > 0)
	{
		delete last_grid_queue_.front();
		last_grid_queue_.pop();
	}
	last_grid_queue_.push(act_grid);
	getOutQueue()->Add(pair<pair<TGrid*, bool>, vector<ImgPose*> >(pair<TGrid*, bool>(act_grid, last_shift), slice_work.first.imgposes_));
	grid_count_++;
	if(last_shift)
		done(true);
}
void GridStage::lastStep()	{ /* omit */ };
