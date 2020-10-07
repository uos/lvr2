#include <iostream>
#include <memory>
#include <tuple>
#include <stdlib.h>
#include <type_traits>
#include <utility>

#include <boost/optional.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// lvr2 includes
#include "lvr2/registration/TransformUtils.hpp"
#include "lvr2/types/MatrixTypes.hpp"

#include "lvr2/io/hdf5/HDF5FeatureBase.hpp"
#include "lvr2/io/hdf5/ArrayIO.hpp"
#include "lvr2/io/hdf5/ChannelIO.hpp"
#include "lvr2/io/hdf5/PointCloudIO.hpp"
#include "lvr2/io/hdf5/MeshIO.hpp"
#include "lvr2/io/hdf5/VariantChannelIO.hpp"
#include "lvr2/io/hdf5/MatrixIO.hpp"
#include "lvr2/io/hdf5/ImageIO.hpp"


#include "lvr2/types/VariantChannel.hpp"
#include "lvr2/types/MatrixTypes.hpp"

cv::Mat generate_image()
{
    cv::Mat img(500, 500, CV_8UC3, cv::Scalar(239,234,224));

    // outter lines
    cv::line(img, cv::Point(250, 50), cv::Point(450, 450), cv::Scalar(0, 0, 0), 3, CV_AA );
    cv::line(img, cv::Point(250, 50), cv::Point(50, 450), cv::Scalar(0, 0, 0), 3, CV_AA );
    cv::line(img, cv::Point(50, 450), cv::Point(450, 450), cv::Scalar(0, 0, 0), 3, CV_AA );

    // inner lines
    cv::line(img, cv::Point(150, 250), cv::Point(350, 250), cv::Scalar(0, 0, 0), 3, CV_AA );
    cv::line(img, cv::Point(150, 250), cv::Point(250, 450), cv::Scalar(0, 0, 0), 3, CV_AA );
    cv::line(img, cv::Point(350, 250), cv::Point(250, 450), cv::Scalar(0, 0, 0), 3, CV_AA );

    return img;
}

void hdf5io_gen_example()
{
    // Empty IO
    using BaseHDF5IO = lvr2::Hdf5IO<>;

    // Extend IO with features (dependencies are automatically fetched)
    using MyHDF5IO = BaseHDF5IO::AddFeatures<
            lvr2::hdf5features::VariantChannelIO,
            lvr2::hdf5features::PointCloudIO,
            lvr2::hdf5features::PointCloudIO // duplicate test
        >;

    // Fast construction
    using MyHDF5IOTest = lvr2::Hdf5Build<lvr2::hdf5features::MeshIO>;

    // Merge two ios
    using MergedIO = MyHDF5IO::Merge<MyHDF5IOTest>;

    // Check if a feature exists in IO Type
    if(MyHDF5IO::HasFeature<lvr2::hdf5features::PointCloudIO>)
    {
        std::cout << "MyHDF5IO has the feature lvr2::hdf5features::PointCloudIO" << std::endl;
    } else {
        std::cout << "MyHDF5IO doesnt have feature lvr2::hdf5features::PointCloudIO" << std::endl;
    }

    // test duplicate feature
    using Duplicate = MyHDF5IO::add_feature<lvr2::hdf5features::PointCloudIO>::type;
    lvr2::Channel<double> channel(1000, 3);
    lvr2::PointBufferPtr pointcloud(new lvr2::PointBuffer);
    pointcloud->add("points", channel);
    Duplicate io;
    io.open("gen_test.h5");
    io.save("apointcloud",pointcloud);

    if(!io.has<lvr2::hdf5features::ImageIO>())
    {
        std::cout << "has feature check on object success" << std::endl;
    }

    // cast
    auto pcl_io = io.scast<lvr2::hdf5features::PointCloudIO>();

    pcl_io->save("bpointcloud", pointcloud);

    lvr2::PointBufferPtr bla = pcl_io->load("bpointcloud");

    auto image_io = io.dcast<lvr2::hdf5features::ImageIO>();

    if(!image_io)
    {
        std::cout << "wrong dynamic cast success" << std::endl;
    }

    auto dyn_pcl_io = io.dcast<lvr2::hdf5features::PointCloudIO>();
    if(dyn_pcl_io)
    {
        std::cout << "correct dynamic cast success" << std::endl;
        dyn_pcl_io->save("cpointcloud", pointcloud);
    }
}

void hdf5io_usage_example()
{
    ///////////////////////
    // Create Some Data  //
    ///////////////////////
    size_t num_data = 10000;
    boost::shared_array<float> data(new float[num_data]);
    lvr2::Channel<double> channel(1000, 3);

    ///////////////////////
    //   Create an IO    //
    ///////////////////////

    using MyHDF5IO = lvr2::Hdf5IO<
        lvr2::hdf5features::ArrayIO,
        lvr2::hdf5features::ChannelIO,
        lvr2::hdf5features::VariantChannelIO,
        lvr2::hdf5features::PointCloudIO,
        lvr2::hdf5features::MeshIO,
        lvr2::hdf5features::MatrixIO,
        lvr2::hdf5features::ImageIO
    >;

    // NEW: Short initialization respecting all dependencies
    using TestIO = lvr2::Hdf5Build<lvr2::hdf5features::MeshIO>;

    MyHDF5IO my_io;
    my_io.open("test.h5");
    
    /////////////////////////
    // 1) ArrayIO Example  //
    /////////////////////////
    my_io.save("agroup", "anarray", num_data, data);
    size_t N;
    boost::shared_array<float> data_loaded = my_io.ArrayIO::load<float>("agroup", "anarray", N);
    
    if(N == num_data)
    {
        std::cout << "ArrayIO successful" << std::endl;
    }

    //////////////////////////
    // 2) ChannelIO Example //
    //////////////////////////
    my_io.save("agroup", "achannel", channel);
    lvr2::ChannelOptional<float> channel_loaded = my_io.ChannelIO::load<float>("agroup", "achannel");
    
    // alternative:
    // channel_loaded = my_io.loadChannel<float>("agroup","achannel");

    if(channel_loaded)
    {
        if(channel_loaded->numElements() == channel.numElements() && channel_loaded->width() == channel.width())
        {
            std::cout << "ChannelIO successful" << std::endl;
        }
    } else {
        std::cout << "channel not found" << std::endl;
    }

    /////////////////////////////
    // 3) PointCloudIO Example //
    /////////////////////////////
    lvr2::PointBufferPtr pointcloud(new lvr2::PointBuffer);
    pointcloud->add("points", channel);
    my_io.save("apointcloud", pointcloud);
    lvr2::PointBufferPtr pointcloud_loaded = my_io.loadPointCloud("apointcloud");
    if(pointcloud_loaded)
    {
        std::cout << "PointCloudIO read success" << std::endl;
        std::cout << *pointcloud_loaded << std::endl;
    } else {
        std::cout << "PointCloudIO read failed" << std::endl;
    }
    
    /////////////////////////////////
    // 4) VariantChannelIO Example //
    /////////////////////////////////

    using VChannel = lvr2::PointBuffer::val_type;

    auto ovchannel 
        = my_io.loadVariantChannel<VChannel>("apointcloud","points");
    
    if(ovchannel) {
        VChannel vchannel = *ovchannel;
        std::cout << "succesfully read VariantChannel: " << vchannel << std::endl;
    } else {
        std::cout << "could not load point from group apointcloud" << std::endl;
    }

    my_io.save("apointcloud", "points2", *ovchannel);

    //////////////////
    // 5) Matrix IO //
    //////////////////

    lvr2::Transformd T = lvr2::Transformd::Identity();

    std::cout << "saving matrix:" << std::endl;
    std::cout << T << std::endl;

    my_io.save("matrices", "amatrix", T);

    auto T2 = my_io.MatrixIO::template load<lvr2::Transformd>("matrices", "amatrix");
    if(T2)
    {
        std::cout << "succesfully loaded Matrix:" << std::endl;
        std::cout << *T2 << std::endl;
    } else {
        std::cout << "could not load matrix!" << std::endl;
    }

    /////////////////
    // 5) Image IO //
    /////////////////

    

    

    
    // FALLBACK
    cv::Mat a = (cv::Mat_<float>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    my_io.save("images", "fallback", a);
    boost::optional<cv::Mat> mat_loaded = my_io.ImageIO::load("images","fallback");

    if(mat_loaded)
    {
        cv::Mat fb_mat = *mat_loaded;
        std::cout << a << std::endl;
        std::cout << fb_mat << std::endl;
    } else {
        std::cout << "fallback doesnt work" << std::endl;
    }

    cv::Mat b = cv::Mat::zeros(50, 50, CV_16SC4);
    my_io.save("images", "fallback2", b);

    cv::Mat c = *my_io.loadImage("images","fallback2");

    if(b.type() == c.type())
    {
        std::cout << "fallback success" << std::endl;
    }

    // RGB
    cv::Mat image = generate_image();

    my_io.save("images", "image", image);
    boost::optional<cv::Mat> image_loaded = my_io.ImageIO::load("images","image");

    if(image_loaded)
    {
        cv::imshow("loaded image", *image_loaded);
        cv::waitKey(0);
    } else {
        std::cout << "could not load image from hdf5 file." << std::endl;
    }

    // GRAYSCALE
    cv::Mat image_gray;
    cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
    my_io.save("images", "image_gray", image_gray);
    boost::optional<cv::Mat> image_gray_loaded = my_io.ImageIO::load("images","image_gray");


    ///////////////////////
    // 6) MeshIO Example //
    ///////////////////////

    // gen io with dependencies
    using MeshIO = lvr2::Hdf5Build<lvr2::hdf5features::MeshIO>;

    std::cout << "N: " << MeshIO::N << std::endl;

    MeshIO mesh_io;
    mesh_io.open("test.h5");

    lvr2::MeshBufferPtr mesh(new lvr2::MeshBuffer);
    mesh->setVertices(lvr2::floatArr(
                          new float[18] {2, 0, 0, 1, 2, 0, 3, 2, 0, 0, 4, 0, 2, 4, 0, 4, 4, 0}
                      ), 6);
    mesh->setFaceIndices(lvr2::indexArray(
                             new unsigned int[12] {0, 1, 2, 1, 3, 4, 2, 4, 5, 1, 4, 2}
                         ), 4);
    mesh_io.save("multimesh/amesh", mesh);
    lvr2::MeshBufferPtr mesh_loaded = mesh_io.loadMesh("multimesh/amesh");
    if(mesh_loaded)
    {
        std::cout << "MeshIO read success" << std::endl;
        std::cout << *mesh_loaded << std::endl;
    } else {
        std::cout << "MeshIO read failed" << std::endl;
    }
}

void update_example() {

    // printing number of features in the io object

    // number of features: 0
    using BaseIO = lvr2::Hdf5IO<>;
    std::cout << BaseIO::N << std::endl;
    
    // MeshIO + Dependency tree -> 4
    using MeshIO = BaseIO::AddFeatures<lvr2::hdf5features::MeshIO>;
    std::cout << MeshIO::N << std::endl;

    // No duplicates, ImageIO with no dependencies -> 5
    using MeshIO2 = MeshIO::AddFeatures<lvr2::hdf5features::MeshIO, lvr2::hdf5features::ImageIO>;
    std::cout << MeshIO2::N << std::endl;
}


int main(int argc, char** argv)
{
    hdf5io_gen_example();
    hdf5io_usage_example();
    update_example();
    return 0;
}
