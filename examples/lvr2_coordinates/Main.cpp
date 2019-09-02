#include <iostream>
#include <memory>
#include <tuple>
#include <stdlib.h>
#include <type_traits>

#include <boost/optional.hpp>

// lvr2 includes
#include "lvr2/registration/TransformUtils.hpp"
#include "lvr2/types/MatrixTypes.hpp"

/// INTERFACES

class BaseVehicle {
public:
    virtual void setBrand(std::string brand) {
        m_brand = brand;
    }

    virtual std::string brand() const
    {
        return m_brand;
    }
protected:
    std::string m_brand;
};

class Drivable {
public:
    virtual void drive(float velocity) = 0;
};

class Steerable {
public:
    virtual void steer(float angle) = 0;
};

/// DRIVE IMPLEMENTATIONS
class AckermannDrive : public Drivable {
public:
    virtual void drive(float velocity)
    {
        std::cout << "Drive Ackermann cinematic " << velocity << " m/s" << std::endl;
    }
};

class DifferentialDrive : public Drivable {
public:
    virtual void drive(float velocity)
    {
        std::cout << "Drive Differential cinematic " << velocity << " m/s" << std::endl;
    }
};

// STEER IMPLEMENTATIONS
class AckermannSteer : public Steerable {
public:
    virtual void steer(float angle)
    {
        std::cout << "Steer Ackermann cinematic " << angle << " rad" << std::endl;
    }
};

class DifferentialSteer : public Steerable {
public:
    virtual void steer(float angle)
    {
        std::cout << "Steer Differential cinematic " << angle << " rad" << std::endl;
    }
};



template<typename ...Tp>
class Vehicle : public BaseVehicle, public Tp... {};

using Car = Vehicle<AckermannDrive, AckermannSteer>;
using Volksbot = Vehicle<DifferentialDrive, DifferentialSteer>;

class Toyota : public Car {
public:
    Toyota(){
        setBrand("Toyota");
    }
};

void drive(Drivable& drivable, float velocity)
{
    drivable.drive(velocity);
}



/// IO

class Channel {

};

class Image {

};

class Hyper {

};

/// Interfaces
class BaseIO {
    virtual void open(std::string filename) = 0;
    virtual void close() = 0;
};

class ChannelIO {
public:
    virtual void save(const Channel& channel) = 0;
    virtual void load(Channel& channel) = 0;
};

class ImageIO {
public:
    virtual void save(const Image& image) = 0;
    virtual void load(Image& image) = 0;
};

class HyperIO {
public:
    virtual void save(const Hyper& hyper) = 0;
    virtual void load(Hyper& hyper) = 0;
};

/// Implementations

class BaseHdf5IO : public BaseIO {
public:
    virtual void open(std::string filename)
    {
        std::cout << "[BaseHdf5IO] open" << std::endl;
        m_filename = filename;
    }

    virtual void close()
    {

    }

    std::string name(){
        return m_filename;
    }

protected:
    std::string m_filename;
};

template<typename Base>
class Hdf5ChannelIO : public ChannelIO {
public:

    virtual void save(const Channel& channel)
    {
        Base* hdf5_file = static_cast<Base*>(this);
        std::cout << "[Hdf5ChannelIO] save channel to " << hdf5_file->name() << std::endl;
    }

    virtual void load(Channel& channel)
    {
        Channel c;
        std::cout << "[Hdf5ChannelIO] load channel" << std::endl;
        channel = c;
    }
};

template<typename Base>
class Hdf5ImageIO : public ImageIO {
public:

    virtual void save(const Image& image)
    {
        // Base* hdf5_file = static_cast<Base*>(this);
        // std::cout << "[Hdf5ImageIO] save image to " << hdf5_file->name() << std::endl;
    }

    virtual void load(Image& image)
    {
        Image im;
        std::cout << "[Hdf5ImageIO] load image" << std::endl;
        image = im;
    }
};

template<template<typename Base> typename ...ComponentT >
class Hdf5IO : public BaseHdf5IO, public ComponentT<Hdf5IO<ComponentT...> >... {
public:


    // template<typename T>
    // void save(const T& object){
    //     // ComponentT<Hdf5IO<ComponentT...> >...::save(object);
    // }
};

using ChannelHdf5IO = Hdf5IO<Hdf5ChannelIO>;

using ImageHdf5IO = Hdf5IO<Hdf5ImageIO>;

using ImageChannelHdf5IO = Hdf5IO<Hdf5ChannelIO, Hdf5ImageIO>;

// class ChannelHdf5IO : public BaseHdf5IO, public Hdf5ChannelIO<ChannelHdf5IO> {

// };



void diamondInheritance()
{
    // Car car = Toyota();
    // std::cout << car.brand() << std::endl;
    // car.drive(5.0);
    // car.steer(1.0);
    // drive(car, 0.2);

    // Volksbot bot;
    // bot.drive(2.0);
    // bot.steer(0.1);

    Channel channel;
    Image image;
    Hyper hyper;
    ChannelHdf5IO io;
    io.open("channel.h5");
    io.save(channel);

    // ImageHdf5IO io2;
    // io2.open("image.h5");
    // io2.save(image);

    // ImageChannelHdf5IO io3;
    // io3.open("channelimage.h5");
    // // io3.save(channel);
    // io3.save(image);

    // Hdf5ChannelIO test_io;
    // test_io.save(channel);
}

class Upper {
public:
    std::string name = "file.h5";
    void save(){}
};

template<typename Base>
class Component1 : public Base {
public:
    using Base::name;
    using Base::save;
    void save(Channel channel) {
        std::cout << "save channel to " << name << std::endl;
    }
};

template<typename Base>
class Component2 : public Base {
public:
    using Base::name;
    using Base::save;
    void save(Image image) {
        std::cout << "save image to " << name << std::endl;
    }
};

template<typename Base>
class Component3 : public Base {
public:
    using Base::name;
    using Base::save;
    void save(Hyper hyper) {
        std::cout << "save hyper to " << name << std::endl;
    }
};

// template<template<typename Base> typename ...ComponentTs>
// class ComponentCollection;

// template<template<typename Base> typename ComponentT>
// class ComponentCollection<ComponentT<Upper> > : public ComponentT<Upper> {

// };

// template<template<typename Base> typename ComponentT, template<typename Base> typename ...ComponentTs>
// class ComponentCollection< : public ComponentT<ComponentCollection<ComponentTs...> > {

// };


struct Parent {

};

template<typename P>
struct Child : public P {

};


template<class ...ComponentTs>
class InheritanceChain {
public:
    InheritanceChain(){std::cout << "last" << std::endl;}
};

// partial specialization at least one template parameter
template <
    class ComponentT,
    class ...ComponentTs
> class InheritanceChain<ComponentT, ComponentTs...> 
: public ComponentT, public InheritanceChain<ComponentTs...> {
public:
    InheritanceChain(){std::cout << "component " << sizeof...(ComponentTs) << std::endl;}
};






void test2()
{
    InheritanceChain<Child<Parent>, Parent> abc;

    Channel channel;
    Image image;
    Hyper hyper;

    Component3<Component2<Component1<Upper> > > bla;

    bla.save(channel);
    bla.save(image);
    bla.save(hyper);

    // ComponentCollection<Component1> bla2;


    
}

int main(int argc, char** argv)
{
    std::cout << "Coordinate Example" << std::endl;
    diamondInheritance();
    test2();
    return 0;
    // Overview Coordinate Systems

    /** LVR / ROS
    *        z  x    
    *        | /
    *   y ___|/ 
    * 
    *  - x: front
    *  - y: left
    *  - z: up
    *  - scale: m
    */

    /** OpenCV
     * 
     *        z    
     *       /
     *      /___ x
     *     |
     *     |
     *     y
     * 
     * - x: right
     * - y: down
     * - z: front
     * - scale: m
     */


    // x: 2.0, y: 0.5, z: 1.0
    // front: 1.0, right: 2.0, down: 0.5
    lvr2::Vector3d cv_point = {2.0, 0.5, 1.0};
    std::cout << "cv point: " << cv_point.transpose() << std::endl;

    // convert to lvr
    // should be x(front): 1.0, y(left): -2.0, z(up): -0.5 
    lvr2::Vector3d lvr_point = lvr2::openCvToLvr(cv_point);
    std::cout << "lvr point: " << lvr_point.transpose() << std::endl;

    if(lvr2::lvrToOpenCv(lvr_point) == cv_point)
    {
        std::cout << "LVR <-> OpenCV - Point: Success" << std::endl;
    }

    // check opencv transformation

    lvr2::Rotationd cv_rot, lvr_rot;

    double roll = -0.25*M_PI; // cv: z, lvr: x
    double pitch = 1.6*M_PI; // cv: -x, lvr: y
    double yaw = -0.06*M_PI; // cv: -y, lvr: z

    // cv rotate x: pitch
    cv_rot = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitZ())
        * Eigen::AngleAxisd(-pitch, Eigen::Vector3d::UnitX())
        * Eigen::AngleAxisd(-yaw, Eigen::Vector3d::UnitY());
    
    // cv -> lvr
    lvr_rot = lvr2::openCvToLvr(cv_rot);


    // cv_rot 

    lvr2::Vector3d cv_point_rotated = cv_rot * cv_point;
    lvr2::Vector3d lvr_point_rotated = lvr_rot * lvr_point;

    // lvr2::openCvToLvr(cv_point_rotated) - lvr_point_rotated
    if((lvr2::openCvToLvr(cv_point_rotated) - lvr_point_rotated).norm() < 0.000001)
    {
        std::cout << "LVR <-> OpenCV - Rotation Matrix: Success" << std::endl;
    } else {
        std::cout << "LVR <-> OpenCV - Rotation Matrix: Wrong" << std::endl;
    }

    // transformation
    lvr2::Transformd cv_transform, lvr_transform;

    cv_transform = lvr2::Transformd::Identity();
    cv_transform.block<3,3>(0,0) = cv_rot;
    cv_transform(0,2) = 2.0;
    cv_transform(1,2) = 5.0;
    cv_transform(2,2) = -1.0;

    lvr_transform = lvr2::openCvToLvr(cv_transform);

    lvr2::Vector3d cv_point_transformed = cv_transform * cv_point;
    lvr2::Vector3d lvr_point_transformed = lvr_transform * lvr_point;

    if((lvr2::openCvToLvr(cv_point_transformed)-lvr_point_transformed).norm() < 0.000001)
    {
        std::cout << "LVR <-> OpenCV - Transformation Matrix: Success" << std::endl;
    } else {
        std::cout << "LVR <-> OpenCV - Transformation Matrix: Wrong" << std::endl;
    }

    return 0;
}