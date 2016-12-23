#pragma once

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz/vizcore.hpp>
#include <kfusion/kinfu.hpp>
#include <lvr/geometry/HalfEdgeVertex.hpp>

#include <io/capture.hpp>

using namespace kfusion;
typedef lvr::ColorVertex<float, unsigned char> cVertex;
typedef lvr::HalfEdgeVertex<cVertex, lvr::Normal<float> >* VertexPtr;

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

        if(event.code == 'c' || event.code == 'C')
            kinfu.checkForShift();

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
        if(!event.symbol.compare("Escape"))
        {
			 kinfu.exit_ = true;
		}
    }
    KinFuApp(OpenNISource& source, Options* options) :
				exit_ (false),  iteractive_mode_(false), pause_(false), meshRender_(false), no_viz_(options->noVizualisation()),
				capture_ (source), cube_count_(0), pic_count_(0), mesh_(NULL), garbageMesh_(NULL)
    {
        KinFuParams params = KinFuParams::default_params();
        params.shifting_distance = options->getShiftingDistance();
        params.distance_camera_target = options->getCameraOffset();
        params.volume_pose = params.volume_pose.translate(Vec3f(0, 0,  options->getCameraOffset()));
        params.cmd_options = options;

        kinfu_ = KinFu::Ptr( new KinFu(params) );

        capture_.setRegistration(true);

		viz.showWidget("legend", cv::viz::WText("Controls", cv::Point(5, 205), 30, cv::viz::Color::red()));
		viz.showWidget("r", cv::viz::WText("r Trigger record", cv::Point(5, 175), 20, cv::viz::Color::green()));
		viz.showWidget("d", cv::viz::WText("d Trigger pause", cv::Point(5, 150), 20, cv::viz::Color::green()));
		viz.showWidget("t", cv::viz::WText("t Finish scan", cv::Point(5, 125), 20, cv::viz::Color::green()));
		viz.showWidget("g", cv::viz::WText("g Export image & pose", cv::Point(5, 100), 20, cv::viz::Color::green()));
		viz.showWidget("i", cv::viz::WText("i Interactive mode", cv::Point(5, 75), 20, cv::viz::Color::green()));
		viz.showWidget("+", cv::viz::WText("+/- Change cam distance", cv::Point(5, 50), 20, cv::viz::Color::green()));
		viz.showWidget("esc", cv::viz::WText("ESC Quit", cv::Point(5, 25), 20, cv::viz::Color::green()));
        cv::viz::WCube cube(cv::Vec3d::all(0), cv::Vec3d(params.volume_size), true, cv::viz::Color::red());
        //cv::viz::WArrow arrow(cv::Point3f(0, 0, 0), cv::Point3f(0, 0, options->getCameraOffset()), 0.03, cv::viz::Color::green());
        cv::Vec2d fov(0.64,0.48);
        cv::viz::WCameraPosition arrow(fov, 0.5, cv::viz::Color::silver());
        cv::viz::WSphere sphere(cv::Point3f(0, 0, 0), params.shifting_distance);
        sphere.setRenderingProperty(cv::viz::OPACITY, 0.2);
        cube.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
        arrow.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
        viz.showWidget("cube0", cube);
        viz.showWidget("arrow", arrow);
        //viz.showWidget("center", sphere);
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
		timer_start_ = (double)cv::getTickCount();
		set_interactive();
		sample_poses_.push_back(kinfu_->getCameraPose());
        viz.showWidget("path", cv::viz::WTrajectory(sample_poses_));
    }

    void show_mesh()
    {
		unordered_map<VertexPtr, size_t> index_map;
		size_t verts_size = 0;
		size_t faces_size = 0;
		while(true)
		{
			auto lvr_mesh = kinfu_->cyclical().getMesh();
			size_t slice_size = lvr_mesh->getVertices().size() - verts_size;
			size_t slice_face_size = lvr_mesh->getFaces().size() - faces_size;
			cv::viz::Mesh* cv_mesh;
			//fill cloud
			cv::Mat verts;
			cv::Vec3f *ddata;
			if(mesh_ == NULL)
			{
				cv_mesh = new cv::viz::Mesh();
				cv_mesh->cloud.create(1, slice_size, CV_32FC3);
				ddata = cv_mesh->cloud.ptr<cv::Vec3f>();
			}
			else
			{
				verts.create(1, slice_size, CV_32FC3);
				ddata = verts.ptr<cv::Vec3f>();
			}

			for(size_t k = 0; k < slice_size; k++)
			{
				auto vertex = lvr_mesh->getVertices()[k + verts_size];
				index_map[vertex] = k + verts_size;
				*ddata++ = cv::Vec3f(vertex->m_position[0], vertex->m_position[1], vertex->m_position[2]);
			}
			if(mesh_ != NULL)
				cv::hconcat(mesh_->cloud, verts, mesh_->cloud);
			//fill polygons
			cv::Mat faces;
			int* poly_ptr;
			if(mesh_ == NULL)
			{
				cv_mesh->polygons.create(1, lvr_mesh->getFaces().size() * 4, CV_32SC1);
				poly_ptr = cv_mesh->polygons.ptr<int>();
			}
			else
			{
				faces.create(1, slice_face_size * 4, CV_32SC1);
				poly_ptr = faces.ptr<int>();
			}
			for(size_t k = 0; k < slice_face_size; k++)
			{
				auto face = lvr_mesh->getFaces()[k + faces_size];
				*poly_ptr++ = 3;
				*poly_ptr++ = index_map[face->m_edge->end()];
				*poly_ptr++ = index_map[face->m_edge->next()->end()];
				*poly_ptr++ = index_map[face->m_edge->next()->next()->end()];
			}
			if(mesh_ != NULL)
				cv::hconcat(mesh_->polygons, faces, mesh_->polygons);

			//fill color
			cv::Mat colors;
			cv::Mat buffer;
			cv::Vec3d *cptr;
			size_t size;
			//auto cBuffer = lvr_mesh->meshBuffer()->getVertexColorArray(size);
			auto fused_map = lvr_mesh->m_fused_verts;
			//cout << "slcie size " << slice_size << endl;
			//cout << "color size " << size << endl;
			if(mesh_ == NULL)
			{
				cv_mesh->colors.create(1, slice_size, CV_64FC(3));
				cptr = cv_mesh->colors.ptr<cv::Vec3d>();
			}
			else
			{
				buffer.create(1, slice_size, CV_64FC(3));
				cptr = buffer.ptr<cv::Vec3d>();
				//buffer.convertTo(colors, CV_8U, 255.0);

			}
			for(size_t i = 0; i < slice_size; ++i)
			{
				if(lvr_mesh->getVertices()[i + verts_size]->m_fused && !lvr_mesh->getVertices()[i + verts_size]->m_fusedNeighbor)
					*cptr++ = cv::Vec3d(0.0, 0.0, 255.0);
				else if(lvr_mesh->getVertices()[i + verts_size]->m_fusedNeighbor)
					*cptr++ = cv::Vec3d(255.0, 0.0, 0.0);
				else
					*cptr++ = cv::Vec3d(0.0, 255.0, 0.0);
			}
			//*cptr++ = cv::Vec3d(3 * cBuffer[fused_map[i]], 3 * cBuffer[fused_map[i] + 1], 3 * cBuffer[fused_map[i] + 2]);
			if(mesh_ != NULL)
			{
				buffer.convertTo(buffer, CV_8U, 255.0);
				cv::hconcat(mesh_->colors, buffer, mesh_->colors);
			}
			else
			{
				cv_mesh->colors.convertTo(cv_mesh->colors, CV_8U, 255.0);
				mesh_ = cv_mesh;
			}
			verts_size = lvr_mesh->getVertices().size();
			faces_size = lvr_mesh->getFaces().size();
			meshRender_ = true;
		}
	}

  void checkForShift()
  {
      kinfu_->triggerCheckForShift();
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
    //if (iteractive_mode_)
    //kinfu.renderImage(view_device_, viz.getViewerPose(), mode);
    //else
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
    viz.setWidgetPose("cube0", kinfu.tsdf().getPose());
    //cv::Affine3f center = kinfu.tsdf().getPose().translate(cv::Vec3f(1.5, 1.5, 1.5));
    //viz.setWidgetPose("center", center);
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
    kinfu.performLastScan();
  }

	void storePicPose(KinFu& kinfu, Affine3f pose, cv::Mat image)
  {
		ImgPose* imgpose = new ImgPose();
		imgpose->pose = pose;
		imgpose->image = image;//.clone();
		//intrinsics wrong? changed rows and cols + /2
		//kinfu.params().cols/2 - 0.5f
		//kinfu.params().rows/2 - 0.5f
		cv::Mat intrinsics = (cv::Mat_<float>(3,3) << kinfu.params().intr.fx*2, 0, 1280/2-0.5f + 3,
												  0, kinfu.params().intr.fx*2, 1024/2-0.5f,
												  0, 0, 1);
		imgpose->intrinsics = intrinsics;
		kinfu.cyclical().addImgPose(imgpose);
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
        cv::Mat depth, image, image_copy;
        double time_ms = 0;
        int has_image = 0;
        std::thread mesh_viz;

        std::vector<Affine3f> posen;
		std::vector<cv::Mat> rvecs;

		Affine3f best_pose;
		cv::Mat best_rvec,best_image;
		float best_dist=0.0;

        if(!no_viz_)
        {
			mesh_viz = thread(&KinFuApp::show_mesh,this);
		}
        for (int i = 0; !exit_ && !viz.wasStopped(); ++i)
        {
			if((!pause_ || !capture_.isRecord()) && !(kinfu.hasShifted() && kinfu.isLastScan()))
            {
				int has_frame = capture_.grab(depth, image);
				cv::flip(depth, depth, 1);
				cv::flip(image, image, 1);
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

            if ((!pause_ || !capture_.isRecord()) && !(kinfu.hasShifted() && kinfu.isLastScan()) && has_image)
            {
				//biggest rvec difference -> new pic
				//
				double ref_timer = cv::getTickCount();
				double time = abs((timer_start_ - ref_timer))/ cv::getTickFrequency();

				if(rvecs.size()<1){
					image.copyTo(image_copy);

					//buffer of all imgposes
					rvecs.push_back(cv::Mat(kinfu.getCameraPose().rvec()));
					posen.push_back(kinfu.getCameraPose());

					//storePicPose(kinfu, image_copy);
					//extractImage(kinfu, image_copy);
				}
                else
                {
					float dist = 0.0;
					cv::Mat mom_rvec(kinfu.getCameraPose().rvec());
					for(size_t z=0;z<rvecs.size();z++){
						dist += norm(mom_rvec-rvecs[z]);
					}
					if(dist > best_dist){
						best_dist = dist;
						//mom_rvec.copyTo(best_rvec);
						//image.copyTo(best_image);
						best_rvec = mom_rvec.clone();
						best_image = image.clone();
						best_pose = kinfu.getCameraPose();
						//std::cout << "better image found, sum rvec distances: " << best_dist << std::endl;
					}
					//if(time - 3.0 > 0)
					if(kinfu.params().cmd_options->textures() && frame_count==7)
					{

						rvecs.push_back(best_rvec);
						posen.push_back(best_pose);

						storePicPose(kinfu, best_pose, best_image);
						//extractImage(kinfu, best_image);
						sample_poses_.push_back(kinfu_->getCameraPose());
						viz.showWidget("path", cv::viz::WTrajectory(sample_poses_));
						std::cout << "image taken "<< image_count++ << ", time: "<< time << std::endl;
						timer_start_ = ref_timer;

					}
				}
                show_raycasted(kinfu);
            }

			if(meshRender_)
			{
				if(garbageMesh_ != NULL)
				{
					viz.removeWidget("mesh");
					//delete garbageMesh_;
				}
				viz.showWidget("mesh", cv::viz::WMesh(*mesh_));
				garbageMesh_ = mesh_;
				meshRender_ = false;
			}

			if(kinfu.hasShifted())
				show_cube(kinfu);

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
				case 'c': case 'C' : checkForShift(); break;
				case '+': kinfu.params().distance_camera_target += 0.1; kinfu.performShift(); break;
				case '-': kinfu.params().distance_camera_target -= 0.1; kinfu.performShift(); break;
				case 27: case 32: exit_ = true; break;
            }
			frame_count++;
			if(frame_count==20) frame_count=0;
            viz.spinOnce(2, true);
			//exit_ = exit_ || ( kinfu.hasShifted() && kinfu.isLastScan() );
			//if(kinfu.cyclical().getSliceCount() == 4 && !(kinfu.hasShifted() && kinfu.isLastScan()))
				//take_cloud(kinfu);
        }
        return true;
    }

	unsigned int image_count=0;
	unsigned int frame_count=0;
    bool exit_, iteractive_mode_, pause_, meshRender_, no_viz_;
    OpenNISource& capture_;
    KinFu::Ptr kinfu_;
    vector<Affine3f> sample_poses_;
    cv::viz::Viz3d viz;
    cv::viz::Mesh* mesh_;
    cv::viz::Mesh* garbageMesh_;
	size_t cube_count_, pic_count_;
	double timer_start_;
    cv::Mat view_host_;
    cv::Mat* image_;
    cuda::Image view_device_;
    cuda::Depth depth_device_;
    cuda::DeviceArray<Point> cloud_buffer;
};
