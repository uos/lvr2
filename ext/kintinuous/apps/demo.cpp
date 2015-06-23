#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz/vizcore.hpp>
#include <kfusion/kinfu.hpp>
#include <io/capture.hpp>

using namespace kfusion;

struct KinFuApp
{
    static void KeyboardCallback(const cv::viz::KeyboardEvent& event, void* pthis)
    {
        KinFuApp& kinfu = *static_cast<KinFuApp*>(pthis);

        if(event.action != cv::viz::KeyboardEvent::KEY_DOWN)
            return;
        
        if(event.code == 't' || event.code == 'T')
            kinfu.take_cloud(*kinfu.kinfu_);

        if(event.code == 'i' || event.code == 'I')
            kinfu.set_interactive();
            
        if(event.code == 'r' || event.code == 'R')
        {
            kinfu.kinfu_->triggerRecord(); 
            kinfu.capture_.triggerRecord();
         }
        
        if(event.code == 'g' || event.code == 'G')
            kinfu.extractImage(*kinfu.kinfu_, *kinfu.image_);
            
        if(event.code == 'd' || event.code == 'D')
        {
            kinfu.capture_.triggerPause();
            kinfu.pause_ = !kinfu.pause_;
        }
        
        if(event.code == '+')
        {			
           kinfu.kinfu_->params().distance_camera_target += 0.1;
           kinfu.kinfu_->performShift();
        }
        if(event.code == '-')
        {			
           kinfu.kinfu_->params().distance_camera_target -= 0.1;
           kinfu.kinfu_->performShift();
        }
    }

    KinFuApp(OpenNISource& source) : exit_ (false),  iteractive_mode_(false), pause_(false),
										capture_ (source), cube_count_(0), pic_count_(0)
    {
        KinFuParams params = KinFuParams::default_params();
        kinfu_ = KinFu::Ptr( new KinFu(params) );

        capture_.setRegistration(true);

		viz.showWidget("legend", cv::viz::WText("Controls", cv::Point(5, 205), 30, cv::viz::Color::red()));
		viz.showWidget("r", cv::viz::WText("r Trigger record", cv::Point(5, 175), 20, cv::viz::Color::green()));
		viz.showWidget("d", cv::viz::WText("d Trigger pause", cv::Point(5, 150), 20, cv::viz::Color::green()));
		viz.showWidget("t", cv::viz::WText("t Finish scan", cv::Point(5, 125), 20, cv::viz::Color::green()));
		viz.showWidget("g", cv::viz::WText("g Export image & pose", cv::Point(5, 100), 20, cv::viz::Color::green()));
		viz.showWidget("i", cv::viz::WText("i Interactive mode", cv::Point(5, 75), 20, cv::viz::Color::green()));
		viz.showWidget("+", cv::viz::WText("+/- Change cam distance", cv::Point(5, 50), 20, cv::viz::Color::green()));
		viz.showWidget("esc", cv::viz::WText("esc Finish and quit", cv::Point(5, 25), 20, cv::viz::Color::green()));
        cv::viz::WCube cube(cv::Vec3d::all(0), cv::Vec3d(params.volume_size), true, cv::viz::Color::red());
        cv::viz::WArrow arrow(cv::Point3f(0, 0, 0), cv::Point3f(0, 0, 0.5), 0.05, cv::viz::Color::green());
        viz.showWidget("cube0", cube);
        viz.showWidget("arrow", arrow);
        viz.showWidget("coor", cv::viz::WCoordinateSystem(0.3));
        //show_cube(*kinfu_);
        viz.setWidgetPose("cube0", kinfu_->tsdf().getPose());
        viz.registerKeyboardCallback(KeyboardCallback, this);
        viz.setWindowSize(cv::Size(1800,480));       
        viz.setWindowPosition(cv::Point(0,0));       
        
        cv::namedWindow("Scene", 0 );
        cv::resizeWindow("Scene",800,500);
        cv::moveWindow("Scene", 800, 500);
        
        cv::namedWindow("Image", 0 );
		cv::resizeWindow("Image",800, 500);
		cv::moveWindow("Image", 0, 500);
        show_mesh();
    }
    
    void show_mesh()
    {
		auto mesh = kinfu_->cyclical().getMesh();
		//cv::viz::Mesh cvmesh = cv::viz::Mesh::load("test_mesh.ply");
		//viz.showWidget("mesh", cv::viz::WMesh(cvmesh));
	}

    void show_depth(const cv::Mat& depth)
    {
        cv::Mat display;
        //cv::normalize(depth, display, 0, 255, cv::NORM_MINMAX, CV_8U);
        depth.convertTo(display, CV_8U, 255.0/4000);
		cv::imshow("Depth", display);
    }

    void show_raycasted(KinFu& kinfu)
    {
        const int mode = 4;
        if (iteractive_mode_)
          kinfu.renderImage(view_device_, viz.getViewerPose(), mode);
        else
			kinfu.renderImage(view_device_, mode);

        view_host_.create(view_device_.rows(), view_device_.cols(), CV_8UC4);
        view_device_.download(view_host_.ptr<void>(), view_host_.step);
        
        cv::imshow("Scene", view_host_);
		
		if(iteractive_mode_)
			viz.setWidgetPose("arrow", kinfu.getCameraPose());  
    }
    
    void show_cube(KinFu& kinfu)
    {
		cube_count_++;
		string new_cube_name = string("cube" + to_string(cube_count_));
		cv::viz::WCube cube(cv::Vec3d::all(0), cv::Vec3d(kinfu.params().volume_size), true, cv::viz::Color::apricot());
		//viz.showWidget(new_cube_name, cube);
		//viz.setWidgetPose(new_cube_name, viz.getWidgetPose("cube0"));
        viz.setWidgetPose("cube0", kinfu.tsdf().getPose()); 
        
	}
	
	void set_interactive()
	{   
		iteractive_mode_ = !iteractive_mode_;
		Affine3f eagle_view = Affine3f::Identity();
		eagle_view.translate(kinfu_->getCameraPose().translation());
		viz.setViewerPose(eagle_view.translate(Vec3f(0,0,-1)));
	}
	
	
    void take_cloud(KinFu& kinfu)
    {
		cout << "Performing last scan" << std::endl;
		
		
		/*Vec3i global_shift_;
		tsdf_buffer buffer_ =  kinfu.cyclical().getBuffer();
        //cuda::DeviceArray<Point> cloud = kinfu.tsdf().fetchCloud(cloud_buffer, kinfu.cyclical().getBuffer());
    	cuda::DeviceArray<Point> cloud = kinfu.tsdf().fetchSliceAsCloud (cloud_buffer, &buffer_, buffer_.voxels_size.x - 1, buffer_.voxels_size.y - 1, buffer_.voxels_size.z - 1, global_shift_);
    	cv::Mat cloud_host(1, (int)cloud.size(), CV_32FC4);
        cloud.download(cloud_host.ptr<Point>());
        cv::Mat colors (cloud_host.size (), CV_8UC3);
         for(int i = 0; i < (int)cloud_host.total(); ++i)
         {
			if(cloud_host.at<Point>(i).w < 0) 
				colors.at<cv::Vec3b>(i) = cv::Vec3b(0, 255, 0);
			else
				colors.at<cv::Vec3b>(i) = cv::Vec3b(255, 0, 0);
          }
        float voxelsize = 3.0 / 512.0;
        for(int i = 0; i < (int)cloud_host.total(); ++i)
        {
			cloud_host.at<Point>(i).x = cloud_host.at<Point>(i).x * voxelsize;
			cloud_host.at<Point>(i).y = cloud_host.at<Point>(i).y * voxelsize;
			cloud_host.at<Point>(i).z = cloud_host.at<Point>(i).z * voxelsize;
		}
        cv::viz::WCloud cloud_widget = cv::viz::WCloud(cloud_host, colors);
        cloud_widget.setRenderingProperty(cv::viz::POINT_SIZE, 2.0);
        cv::viz::writeCloud("cloud.ply", cloud_host, colors);*/
        //viz.showWidget("cloud", cv::viz::WPaintedCloud(cloud_host));
		kinfu.performLastScan();
    }
    void extractImage(KinFu& kinfu, cv::Mat& image)
    {
		pic_count_++;
		auto trans = kinfu.getCameraPose().translation();
		auto rot =  kinfu.getCameraPose().rotation();
		imwrite( string("Pic" + std::to_string(pic_count_) + ".png"), image );
		ofstream pose_file;
		pose_file.open (string("Pic" + std::to_string(pic_count_) + ".txt"));
		pose_file << "Translation: " << endl;
		pose_file << trans[0] << endl;
		pose_file << trans[1] << endl;
		pose_file << trans[2] << endl;
		pose_file << endl;
	    pose_file << "Rotation: " << endl;
	    pose_file << rot(0,0) << " " << rot(0,1) << " " << rot(0,2) << endl;
	    pose_file << rot(1,0) << " " << rot(1,1) << " " << rot(1,2) << endl;
	    pose_file << rot(2,0) << " " << rot(2,1) << " " << rot(2,2) << endl;
	    pose_file << endl;
	    pose_file << "Camera Intrinsics: " << endl;
	    pose_file << kinfu.params().intr.fx << " " <<  kinfu.params().rows << " " << kinfu.params().cols << endl;
	    pose_file.close();
	}
		

    bool execute()
    {
        KinFu& kinfu = *kinfu_;
        cv::Mat depth, image;
        double time_ms = 0;
        int has_image = 0;

        for (int i = 0; !exit_ && !viz.wasStopped(); ++i)
        {
			if(!pause_ || !capture_.isRecord())
            {
				int has_frame = capture_.grab(depth, image);
				image_ = &image;
				if (has_frame == 0)
					return std::cout << "Can't grab" << std::endl, false;
				// check if oni file ended
				if (has_frame == 2)
					take_cloud(kinfu);
				depth_device_.upload(depth.data, depth.step, depth.rows, depth.cols);

				{
					SampledScopeTime fps(time_ms); (void)fps;
					has_image = kinfu(depth_device_);
				}
			}

            if (has_image)
                show_raycasted(kinfu);
                
			
			if(kinfu.hasShifted())
				show_cube(kinfu);
            //show_depth(depth);

			cv::imshow("Image", image);

            if (!iteractive_mode_)
            {
                viz.setViewerPose(kinfu.getCameraPose());
			}
				
            int key = cv::waitKey(3);

            switch(key)
            {
				case 't': case 'T' : take_cloud(kinfu); break;
				case 'i': case 'I' : set_interactive(); break;
				case 'd': case 'D' : capture_.triggerPause();pause_  = !pause_; break;
				case 'r': case 'R' : kinfu.triggerRecord(); capture_.triggerRecord(); break;
				case 'g': case 'G' : extractImage(kinfu, image); break;
				case '+': kinfu.params().distance_camera_target += 0.1; kinfu.performShift(); break;
				case '-': kinfu.params().distance_camera_target -= 0.1; kinfu.performShift(); break;
				case 27: case 32: exit_ = true; break;
            }

            viz.spinOnce(3, true);
			exit_ = exit_ || ( kinfu.hasShifted() && kinfu.isLastScan() );
			//if(kinfu.cyclical().getSliceCount() == 14)
				//take_cloud(kinfu);
        }
        return true;
    }

    bool exit_, iteractive_mode_, pause_;
    OpenNISource& capture_;
    KinFu::Ptr kinfu_;
    cv::viz::Viz3d viz;
	size_t cube_count_, pic_count_;
    cv::Mat view_host_;
    cv::Mat* image_;
    cuda::Image view_device_;
    cuda::Depth depth_device_;
    cuda::DeviceArray<Point> cloud_buffer;
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main (int argc, char* argv[])
{
    int device = 0;
    cuda::setDevice (device);
    cuda::printShortCudaDeviceInfo (device);

    if(cuda::checkIfPreFermiGPU(device))
        return std::cout << std::endl << "Kinfu is not supported for pre-Fermi GPU architectures, and not built for them by default. Exiting..." << std::endl, 1;

	OpenNISource capture;
	
	if(argc == 2)
		capture.open(string(argv[1]));
	else
	{
		capture.open(0);
	}
   
    //capture.open (0);
    //capture.open("/home/tristan/kintinuous.tigelbri/build/Captured.oni");
    //capture.open("/home/tristan/home.oni");

    KinFuApp app (capture);

    // executing
    try { app.execute (); }
    catch (const std::bad_alloc& /*e*/) { std::cout << "Bad alloc" << std::endl; }
    catch (const std::exception& /*e*/) { std::cout << "Exception" << std::endl; }

    return 0;
}
