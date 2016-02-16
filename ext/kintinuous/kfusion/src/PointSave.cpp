#include <kfusion/PointSave.hpp>

using namespace lvr;

bool PointSave::savePoints(const char* filename,std::vector<std::vector<ColorVertex<float, unsigned char> > > rcb,int size){
	if(rcb.size() > 0) {
		ofstream point_file;
		point_file.open(filename);
		
		point_file << "ply" << endl;
		point_file << "format ascii 1.0" << endl;
		point_file << "element vertex "<< std::to_string(size)<< endl;
		point_file << "property float x\nproperty float y\nproperty float z" << endl;
		point_file << "property uchar red\nproperty uchar green\nproperty uchar blue" << endl;

		point_file << "end_header" << endl;
	
		for(int i=0;i<rcb.size();i++){
			for(int j=0;j<rcb[i].size();j++){
				for(int k=0;k<3;k++){
					point_file << rcb[i][j][k] << " ";
				}
				point_file << i*255/rcb.size() << " " << i*255/rcb.size() << " " << i*255/rcb.size() << endl;
				
			}
		}
		
		point_file.close();
	}
}

bool PointSave::savePointsWithCameras(const char* filename,std::vector<std::vector<ColorVertex<float, unsigned char> > > rcb, int size, std::vector<kfusion::ImgPose> image_pose_buffer){
	if(rcb.size() > 0) {
		ofstream point_file;
		point_file.open(filename);
		
		point_file << "ply" << std::endl;
		point_file << "format ascii 1.0" << std::endl;
		point_file << "element vertex "<< std::to_string(size +image_pose_buffer.size()) << std::endl;
		point_file << "property float x\nproperty float y\nproperty float z" << std::endl;
		point_file << "property uchar red\nproperty uchar green\nproperty uchar blue" << std::endl;


		point_file << "end_header" << std::endl;
	
		for(int i=0;i<rcb.size();i++){
			for(int j=0;j<rcb[i].size();j++){
				for(int k=0;k<3;k++){
					point_file << rcb[i][j][k] << " ";
				}
				point_file << i*255/rcb.size() << " " << i*255/rcb.size() << " " << i*255/rcb.size() << std::endl;
			}
		}
		
		//cameras einzeichnen in rot
		cv::Vec3f dst;

		for(int i=0;i<image_pose_buffer.size();i++){
			cv::Vec3f z_dir(image_pose_buffer[i].pose.rotation() * (cv::Vec3f(0,0,1)));
			z_dir = z_dir * 1/4;
			dst = image_pose_buffer[i].pose.translation();
			point_file << dst[0] << " " << dst[1] << " " << dst[2] << " 255 0 0"<< endl;
			cv::Vec3f direction = dst + z_dir;
			point_file << dst[0] << " " << dst[1] << " " << dst[2] << " 0 255 0"<< endl;
		}
	
		point_file.close();
	}
}

bool PointSave::saveBoundingRectangles(const char* filename, std::vector< std::vector<cv::Point3f> > b_rects, int size){
	if(b_rects.size() > 0) {
			ofstream point_file;
			point_file.open(filename);

			point_file << "ply" << std::endl;
			point_file << "format ascii 1.0" << std::endl;
			point_file << "element vertex "<<to_string(size)<< std::endl;
			point_file << "property float x\nproperty float y\nproperty float z" << std::endl;
			point_file << "property uchar red\nproperty uchar green\nproperty uchar blue" << std::endl;
			//optional
			point_file << "element face " << to_string(size/4) << std::endl;
			point_file << "property list uchar int vertex_index" << std::endl;

			point_file << "end_header" << std::endl;

			for(int i=0;i<b_rects.size();i++){
				for(int j=0;j<b_rects[i].size();j++){

						point_file << b_rects[i][j].x << " " << b_rects[i][j].y << " " << b_rects[i][j].z << " " ;

						point_file << i*255/b_rects.size() << " " << i*255/b_rects.size() << " " << i*255/b_rects.size() << std::endl;
				}
			}

			for(int i=0;i<size;i+=4){
				point_file << "4 " << i << " " << i+1 << " " << i+2 << " " << i+3 << std::endl;
			}

			std::cout << "saved BoundingBoxes to " << filename << std::endl;
			
			point_file.close();
		}else{
			std::cout << "Couldn't save. Bounding Rects = 0" << std::endl;
		}
}
