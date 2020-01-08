/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <chrono>
#include <iostream>
#include <math.h>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include <boost/filesystem/fstream.hpp>

#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/io/RxpIO.hpp"
#include "lvr2/util/Util.hpp"
#include "lvr2/types/MatrixTypes.hpp"
#include "lvr2/registration/TransformUtils.hpp"

#include "Options.hpp"
#include "RieglProject.hpp"

/*

    TODO:
            If RxpIO can't read the file, it probably gives back an empty modelptr. This case
            crashes the program.


*/

using Vec = lvr2::BaseVector<float>;

void transformModel(lvr2::ModelPtr model, const lvr2::Transformd& transform)
{
    size_t num_points = model->m_pointCloud->numPoints();

    // float test = model->m_pointCloud->getPointArray().get()[0];

    lvr2::BaseVector<float>* pts_raw = 
        reinterpret_cast<lvr2::BaseVector<float>*>(&model->m_pointCloud->getPointArray().get()[0]);

    #pragma omp for
    for(size_t i=0; i<num_points; i++)
    {
        pts_raw[i] = transform * pts_raw[i];
    }
}

template <typename ValueType>
bool write_params_to_file(fs::path dest, bool force_overwrite, ValueType *values, int count) {

    if (!force_overwrite && fs::exists(dest)) {
        std::cout << "[write_params_to_file] Info: Skipping writing to file " << dest << " because the file already exists. (If already existing files should be overwritten, use --force.)" << std::endl;
        return true;
    }

    fs::ofstream out(dest);

    if (!out.is_open() || !out.good()) {
        std::cout << "[write_params_to_file] Error: Unable to write to file " << dest << std::endl;
        return false;
    }


    for (int i = 0; i < count; i++) {
        out << std::setprecision(std::numeric_limits<float>::digits10 + 1) << values[i] << " ";
    }

    out.close();

    return true;
}

int char_to_int(char in) {
    return ((int) in) - 48;
}

template <typename T>
bool write_mat4_to_file(lvr2::Transformd mat, fs::path dest, bool force_overwrite) {

    if (!force_overwrite && fs::exists(dest)) {
        std::cout << "[write_matrix4_to_file] Info: Skipping writing to file " << dest << " because the file already exists. (If already existing files should be overwritten, use --force.)" << std::endl;
        return true;
    }

    fs::ofstream out(dest);

    if (!out.is_open() || !out.good()) {
        std::cout << "[write_matrix4_to_file] Error: Unable to write to file " << dest << std::endl;
        return false;
    }

    for (int i = 0; i < 16; i++) {
        out << std::setprecision(std::numeric_limits<T>::digits10 + 1 ) << mat(i);
        out << (i%4 != 3 || i == 0 ? ' ' : '\n');
    }

    out.close();

    return true;
}

bool copy_file(const fs::path from, const fs::path to, bool force_overwrite) {

        if (!force_overwrite && fs::exists(to)) {
            std::cout << "[copy_file] Info: Skipping copy from file " << from << " because the file " << to << "already exists. (If already existing files should be overwritten, use --force.)" << std::endl;
            return true;
        }

        fs::copy_option co = fs::copy_option::overwrite_if_exists;

        try {
            fs::copy_file(from, to, co);
        } catch (fs::filesystem_error &e) {
            std::cout << "[convert_riegl_project] Error: " << e.what() << std::endl;
            return false;
        }

        return true;
}

bool write_mat4_to_pose_file(const fs::path file, const lvr2::Transformd& transform, bool force_overwrite) {

        if (!force_overwrite && fs::exists(file)) {
            std::cout << "[write_mat4_to_pose_file] Info: Skipping writing " << file << " because it already exists. (If already existing files should be overwritten, use --force.)" << std::endl;
            return true;
        }

        double pose_data[6];

        //lvr2::Util::riegl_to_slam6d_transform(transform).toPostionAngle(pose_data);

        lvr2::Transformd trans_slam6d = lvr2::lvrToSlam6d(transform);

        lvr2::eigenToEuler(trans_slam6d, pose_data);

        // .pose expects angles in degree and not radian
        pose_data[3] = lvr2::Util::rad_to_deg(pose_data[3]);
        pose_data[4] = lvr2::Util::rad_to_deg(pose_data[4]);
        pose_data[5] = lvr2::Util::rad_to_deg(pose_data[5]);

        fs::ofstream pose(file);
        if (!pose.is_open() || !pose.good()) {
            std::cout << "[writing_mat4_to_pose_file] Error: Error while writing " << file << std::endl;
            return false;
        }

        for (int i = 0; i < 6; i++) {
            pose << pose_data[i];

            if (i == 2 || i == 5) {
                pose << '\n';
            } else {
                pose << ' ';
            }
        }

        pose.close();

        return true;
}

void convert_rxp_to_3d_per_thread(
    std::vector<lvr2::ScanPosition> *work,
    int *read_file_count,
    int *current_file_idx,
    const fs::path *scans_dir,
    std::mutex *mtx,
    int id,
    bool force_overwrite,
    unsigned int reduction,
    std::string inputformat = "rxp",
    std::string outputcoords = "slam6d")
{

    lvr2::RxpIO rxpio;

    mtx->lock();

    while (*current_file_idx < work->size()) {

        int scan_nr = *current_file_idx + 1;
        lvr2::ScanPosition &pos = (*work)[(*current_file_idx)++];

        mtx->unlock();

        // build output filename to check if it already exists...
        char out_file_buf[2048];
        std::snprintf(out_file_buf, 2048, "scan%.3d.3d", scan_nr);
        fs::path out_file = *scans_dir / out_file_buf;
        if (!force_overwrite && fs::exists(out_file)) {
            mtx->lock();
            std::cout << "[read_rxp_per_thread] Info: Skipping conversion from file " << pos.scan_file << " because the file " << out_file << "already exists. (If already existing files should be overwritten, use --force.)" << std::endl;
            (*read_file_count)++;
            continue;
        }

        lvr2::Transformd identity;
        lvr2::Transformd riegl_to_slam_transform;

        riegl_to_slam_transform(4)  = -100.0;
        riegl_to_slam_transform(9)  =  100.0;
        riegl_to_slam_transform(5)  =  0.0;
        riegl_to_slam_transform(2)  =  100.0;
        riegl_to_slam_transform(10) =  0.0;

        lvr2::ModelPtr tmp;
        
        if(inputformat == "rxp")
        {
            if(outputcoords == "slam6d")
            {
                tmp = rxpio.read(
                    pos.scan_file.string(),
                    reduction,
                    riegl_to_slam_transform
                );

            } else if(outputcoords == "lvr") {
                tmp = rxpio.read(
                    pos.scan_file.string(),
                    reduction,
                    identity
                );
            }
        } else {
            // ascii etc
            

            lvr2::Transformd inv_transform = pos.transform.inverse();
            inv_transform.transpose();
            
            tmp = lvr2::ModelFactory::readModel(pos.scan_file.string());

            // ascii export is already transformed, have to transform it back.
            // or find the button to save the ascii clouds without transformation
            transformModel(tmp, inv_transform);

            if(outputcoords == "slam6d")
            {
                // convert to slam6d
                std::cout << "[read_rxp_per_thread] Transform to slam6d" << std::endl;
                transformModel(tmp, riegl_to_slam_transform);
            }
        }
        
        lvr2::ModelFactory::saveModel(tmp, out_file.string());

        mtx->lock();
        std::cout << "[read_rxp_per_thread] Info: Thread " << id << ": read data from file " << pos.scan_file << " and wrote to file " << out_file << " ("
                  << (*read_file_count)++ + 1 << "/" << work->size() << ")" << std::endl;
    }

    mtx->unlock();
}

bool convert_riegl_project(
    lvr2::RieglProject &ri_proj,
    const fs::path &out_scan_dir,
    bool force_overwrite,
    unsigned int reduction,
    std::string output_coords = "slam6d"
) {
    //sub folders
    fs::path scans_dir        = out_scan_dir / "scans/";
    fs::path images_dir       = out_scan_dir / "images/";

    // @TODO maybe don't create dirs if nothing will be stored in them...
    // create directory structure
    try {
        fs::create_directory(out_scan_dir);

        fs::create_directory(scans_dir);
        fs::create_directory(images_dir);

    } catch (fs::filesystem_error &e) {
       std::cout << "[convert_riegl_project] Error: " << e.what() << std::endl;
       return false;
    }

    // read pointclouds

    std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();

    // get number of cores...
    unsigned int num_cores = std::thread::hardware_concurrency();
    if (!num_cores) {
        num_cores = 1;
    }

    std::mutex mtx;
    std::thread *threads = new std::thread[num_cores];
    std::vector<lvr2::ModelPtr> clouds;

    int read_file_count = 0;
    int current_file_idx = 0;

    for (int j = 0; j < num_cores; j++) {
        threads[j] = std::thread(
            convert_rxp_to_3d_per_thread,
            &ri_proj.m_scan_positions,
            &read_file_count,
            &current_file_idx,
            &scans_dir,
            &mtx, 
            j,
            force_overwrite,
            reduction,
            ri_proj.m_input_cloud_format,
            output_coords
        );
    }

    for (int j = 0; j < num_cores; j++) {
        threads[j].join();
    }

    delete[] threads;

    std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
    std::chrono::seconds diff = std::chrono::duration_cast<std::chrono::seconds>(end - start);

    std::cout << "Reading " << read_file_count << " files took " << diff.count() << " sec." << std::endl;

    int scan_nr = 1;

    for (lvr2::ScanPosition &pos : ri_proj.m_scan_positions) {

        // copy pose files
        char out_file_buf[2048];
        std::snprintf(out_file_buf, 2048, "scan%.3d.pose", scan_nr);
        fs::path out_pose_file = scans_dir / out_file_buf;
        if (!write_mat4_to_pose_file(out_pose_file, pos.transform, force_overwrite)) {
            std::cout << "[convert_riegl_project] Error: Error while trying to write file " << out_pose_file << std::endl;
            return false;
        }

        int image_nr = 1;
        for (const lvr2::ImageFile &image : pos.images) {
            //copy image files
            char out_file_buf[2048];
            std::snprintf(out_file_buf, 2048, "scan%.3d_%.2d.jpg", scan_nr, image_nr);
            fs::path out_img_file = images_dir / out_file_buf;

            if (!copy_file(image.image_file, out_img_file, force_overwrite)) {
                std::cout << "[convert_riegl_project] Error: Error while trying to copy file "
                          << image.image_file << " to " << out_img_file << std::endl;
                return false;
            }

            //write out extrinsic transform
            std::snprintf(out_file_buf, 2048, "scan%.3d_%.2d_extrinsic.dat", scan_nr, image_nr);
            if (!write_mat4_to_file<double>(image.extrinsic_transform,
                                    images_dir / out_file_buf,
                                    force_overwrite)) {
                std::cout << "[convert_riegl_project] Error: Error while writing image extrinsic \
                          matrix to " << (images_dir / out_file_buf) << std::endl;
            }

            //write out orientation transform
            std::snprintf(out_file_buf, 2048, "scan%.3d_%.2d_orientation.dat", scan_nr, image_nr);
            if (!write_mat4_to_file<double>(image.orientation_transform,
                                    images_dir / out_file_buf,
                                    force_overwrite)) {
                std::cout << "[convert_riegl_project] Error: Error while writing image orientation \
                          matrix to " << (images_dir / out_file_buf)<< std::endl;
            }

            //write out intrinsic params
            std::snprintf(out_file_buf, 2048, "scan%.3d_%.2d_intrinsic.txt", scan_nr, image_nr);
            if (!write_params_to_file(images_dir / out_file_buf,
                                      force_overwrite,
                                      image.intrinsic_params,
                                      4)) {
                std::cout << "[convert_riegl_project] Error: Error while writing image intrinsic \
                          params to " << (images_dir / out_file_buf) << std::endl;
            }

            //write out distortion params
            std::snprintf(out_file_buf, 2048, "scan%.3d_%.2d_distortion.txt", scan_nr, image_nr);
            if (!write_params_to_file(images_dir / out_file_buf,
                                      force_overwrite,
                                      image.distortion_params,
                                      6)) {
                std::cout << "[convert_riegl_project] Error: Error while writing image distortion \
                          params to " << (images_dir / out_file_buf) << std::endl;
            }

            image_nr++;
        }

        scan_nr++;
    }

    return true;
}

int main(int argc, char **argv) {
    // @TODO move other informations into structure (calibrations)
    Options opt;
    if (!opt.parse(argc, argv)) {
        return 0;
    }

    std::cout << "Arguments:\n InputDir: " << opt.getInputDir() << "; OutputDir: " << opt.getOutputDir() << "; Forced: " << opt.force_overwrite() << "; Reduction: " << opt.getReductionFactor() << "; Start: " << opt.getStartscan() << "; End: " << opt.getEndscan() << std::endl;

#if 1
    lvr2::RieglProject tmp(opt.getInputDir(), opt.getInputFormat());

    

    if (!tmp.parse_project(opt.getStartscan(), opt.getEndscan())) {
        std::cout << "[main] Error: The directory \'" << opt.getInputDir() << "\' is NOT a Riegl Scanproject directory." << std::endl;

        return 0;
    }

    if (!convert_riegl_project(tmp, fs::path(opt.getOutputDir()), opt.force_overwrite(), opt.getReductionFactor())) {
            std::cout << "[main] Error: It occured an error while converting the Riegl Scan Project." << std::endl;

        return 0;
    }
#else
    fs::path scan_dir(opt.getInputDir());
    scan_dir = scan_dir / "scans/";
    std::regex files_3d_reg("scan\\d{3}.3d");
    std::regex files_4x4_reg("scan\\d{3}.4x4");

    std::vector<fs::path> cloud_files;
    std::vector<fs::path> transform_files;

    fs::directory_iterator it_begin = fs::directory_iterator(scan_dir);
    fs::directory_iterator it_end   = fs::directory_iterator();
    for (auto it = it_begin; it != it_end; it++) {

        fs::path current_file = *it;

        if (std::regex_match(current_file.filename().string(), files_3d_reg)) {
            cloud_files.push_back(current_file);
        }

        if (std::regex_match(current_file.filename().string(), files_4x4_reg)) {
            transform_files.push_back(current_file);
        }
    }

    std::sort(cloud_files.begin(), cloud_files.end(), [](const fs::path &a, const fs::path &b) { return a.compare(b) < 0; });
    std::sort(transform_files.begin(), transform_files.end(), [](const fs::path &a, const fs::path &b) { return a.compare(b) < 0; });

    std::vector<lvr2::ModelPtr> clouds;
    std::vector<lvr2::Matrix4<Vec>> transforms;

    for (int i = 0; i < cloud_files.size(); i++) {
        clouds.push_back(lvr2::ModelFactory::readModel(cloud_files[i].string()));
        lvr2::Matrix4<Vec> trans;
        trans.loadFromFile(transform_files[i].string());
        trans.transpose();
        transforms.push_back(trans);
    }

    for (int i = 0; i < clouds.size(); i++) {
        if (!clouds[i]->m_pointCloud)
            continue;
        size_t n;
        lvr2::floatArr cloud = clouds[i]->m_pointCloud->getPointArray(n);
        std::cout << n << std::endl;
        for (int j = 0; j < n; j++) {
            lvr2::Vector<Vec> vert(cloud[j*3], cloud[j*3 + 1], cloud[j*3 + 2]);
            lvr2::Vector<Vec> vert2(vert.z/100.0, -vert.x/100.0, vert.y/100.0);
            vert2 = transforms[i] * vert2;
            vert2.transformRM(transforms[i]);
            cloud[j*3] = -100*vert2.y;
            cloud[j*3+1] = 100*vert2.z;
            cloud[j*3+2] = 100*vert2.x;

        }

        lvr2::ModelFactory::saveModel(clouds[i], scan_dir.string() + "trans" + std::to_string(i) + ".3d");
    }

#endif
    return 0;
}
