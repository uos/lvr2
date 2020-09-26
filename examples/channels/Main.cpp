#include <iostream>

#include "lvr2/types/Channel.hpp"
#include "lvr2/types/VariantChannel.hpp"
#include "lvr2/types/VariantChannelMap.hpp"
#include "lvr2/types/MultiChannelMap.hpp"

#include "lvr2/types/BaseBuffer.hpp"

#include "lvr2/algorithm/BaseBufferManipulators.hpp"


using namespace lvr2;

template<typename T>
void fillChannel(Channel<T>& channel, T value)
{
    for(int i=0; i<channel.numElements(); i++)
    {
        for(int j=0; j<channel.width(); j++)
        {
            channel[i][j] = value;
        }
    }
}

/**
 * @brief Basic Channel Usage
 */
void basicChannelUsage()
{
    std::cout << "1) Basic Channel Test" << std::endl;
    size_t num_points = 10000;

    ////////////////
    /// 1) Channels
    //////////////////

    // construct channel objects
    Channel<float> points(num_points, 3);
    Channel<float> normals(num_points, 3);
    Channel<unsigned char> colors(num_points, 3);

    // fill channels
    fillChannel(points, 5.0f);
    fillChannel(normals, 1.0f);
    fillChannel(colors, static_cast<unsigned char>(255));

    // print channels
    std::cout << "  points: " << points << std::endl;
    std::cout << "  number of normals: " << normals.numElements() << std::endl;
    std::cout << "  number of color channels: " << colors.width() << std::endl;

    // copy channels
    std::cout << "  deep copy" << std::endl;
    Channel<float> points2(points);
    std::cout << "  shallow copy" << std::endl;
    Channel<float> points3(0,0);
    points3 = points2;

    /////////////
    /// 2) Variant Channels
    /////////////

    // build a variant channel of two possible types float or unsigned char
    std::cout << "  generate a variant channel..." << std::endl;
    VariantChannel<float, unsigned char> vchannel(points);
    std::cout << "  a variant channel: " << vchannel << std::endl;
    
    // you can fetch the data with dataPtr<Type>()
    auto data = vchannel.dataPtr<float>();
    std::cout << "  first element of points: " << data[0] << std::endl;

    // you can get the index of the variant with this function
    std::cout << "  float is the index " 
        << VariantChannel<float, unsigned char>::index_of_type<float>::value 
        << " in [float, unsigned char]"
        << std::endl;
}


void multiChannelMapUsage()
{
    std::cout << "2) Multi Channel Map Usage" << std::endl;
    size_t num_points = 10000;
    Channel<float> points(num_points, 3);
    Channel<float> normals(num_points, 3);
    Channel<unsigned char> colors(num_points, 3);

    // fill channels
    fillChannel(points, 5.0f);
    fillChannel(normals, 1.0f);
    fillChannel(colors, static_cast<unsigned char>(255));

    ///////////
    /// MultiChannelMap


    

    // 1) Insert Channels of different types
    MultiChannelMap cm = {
        {"points", points}
    };
    cm["colors"] = colors;
    cm.insert({"normals", normals});

    // 2) Information
    std::cout << "  number of channels in channelmap: " << cm.size() << std::endl;
    std::cout << "  number of points: " << cm["points"].numElements() << std::endl;

    

    // 3) Get Channels
    // use only if you now the channel exists
    Channel<float> pts = cm.get<float>("points");
    
    // use this if you dont know
    Channel<float>::Optional not_existing_channel = cm.getOptional<float>("hello world");
    if(not_existing_channel) {
        Channel<float> existing_channel = *not_existing_channel;
    } else {
        std::cout << "  the channel 'hello world' doesnt exist" << std::endl;
    }

    // or use find
    auto it = cm.find("points");
    if(it != cm.end())
    {
        std::cout << "  " << it->first << " found!" << std::endl;
        if(it->second.is_type<float>())
        {
            Channel<float> pts2 = it->second.extract<float>();
        }
    }

    // 4) iterator
    
    std::cout << "  lets iterate" << std::endl;
    for(auto it: cm)
    {
        std::cout << "  --"<< it.first << ": " << it.second.numElements() << " x " << it.second.width() << std::endl;
    }

    MultiChannelMap fcm;

    // typed iteration
    std::cout << "  lets iterate over the type float" << std::endl;
    for(auto it = cm.typedBegin<float>(); it != cm.end(); ++it)
    {
        Channel<float> ch = it->second;
        std::cout << "  --" << it->first << ": " << ch << std::endl;

        fcm.insert(*it);
    }

    // 5) Remove
    
    // remove all floats
    std::cout << "  removing floats..." << std::endl,
    std::cout << "  size before remove: " << cm.size() << std::endl;
    auto it2 = cm.typedBegin<float>();
    while(it2 != cm.end())
    {
        it2 = cm.erase(it2);
    }
    std::cout << "  size after remove: " << cm.size() << std::endl;

    // remove with string
    fcm.erase("points");

    // 6) other functions
    std::vector<std::string> keys = fcm.keys<float>();
    size_t num_float_channels = fcm.numChannels<float>();

    // 7) print
    std::cout << "  The channel map: " << std::endl;
    std::cout << cm << std::endl;

    // 8) copy
    std::cout << "  Channel Map Copy:" << std::endl;
    // deep copy
    std::cout << "  deep copy" << std::endl;
    MultiChannelMap cm2(fcm);
    std::cout << "  deep copy" << std::endl;
    MultiChannelMap cm3 = fcm;
    // shallow copy
    std::cout << "  shallow copy" << std::endl;
    MultiChannelMap cm4;
    cm4 = cm3;
    
}

void channelManagerUsage()
{
    size_t num_points = 10000;

    Channel<float> points(num_points, 3);
    Channel<float> normals(num_points, 3);
    Channel<unsigned char> colors(num_points, 3);
    Channel<unsigned char> hyper(600, 600);
    
    fillChannel(points, 0.0f);
    fillChannel(normals, 1.0f);
    fillChannel(colors, static_cast<unsigned char>(255));
    fillChannel(hyper, static_cast<unsigned char>(100));

    std::cout << "3) Channel Manager Usage" << std::endl;

    // MultiChannelMap with extended functions -> ChannelManager

    BaseBuffer cm = {
        {"points2" , points},
        {"hyper", hyper}
    };
    
    cm["points"] = points;
    cm["colors"] = colors;
    cm["normals"] = normals;


    // Get Channels
    FloatChannelOptional points_test;

    points_test = cm.getFloatChannel("points");
    points_test = cm.getFloatChannel("colors");
    points_test = cm.getFloatChannel("asdf");

    cm.getChannel("points", points_test);
    cm.getChannel("colors", points_test);
    cm.getChannel("asdf", points_test);

    // Atomics
    cm.addFloatAtomic(5.5, "myatomic");
    auto myatomic = cm.getFloatAtomic("myatomic");
    myatomic = cm.getFloatAtomic("bla");

    std::cout << "  total number of elements: " << cm.size() << std::endl;
    std::cout << "  num float channels: " << cm.numChannels<float>() << std::endl;
    std::cout << "  float channels are:" << std::endl;

    for(auto key : cm.keys<float>())
    {
        std::cout << "  -- " << key << std::endl;
    }

    BaseBuffer cm2;

    std::cout << "  float channels again:" << std::endl;
    for(auto it = cm.typedBegin<float>(); it != cm.end(); ++it)
    {
        std::cout << "  -- " << it->first << " " << it->second.numElements() << std::endl;
        cm2.insert({it->first, it->second});
        cm2[it->first] = it->second;
        cm2.insert(*it);
    }
    std::cout << "  inserted elements: " << cm2.size() << std::endl;

    std::cout << "  float iteration with remove:" << std::endl;
    auto it = cm.typedBegin<float>();
    while(it != cm.end())
    {
        std::cout << "  remove " << it->first << std::endl;
        it = cm.erase(it);
    }

    
    std::cout << "  unsigned char iteration:" << std::endl;
    for(auto it = cm.typedBegin<unsigned char>(); it != cm.end(); ++it)
    {
        std::cout << "  -- " << it->first << std::endl;
    }

    UCharChannelOptional colors3 =  cm.getOptional<unsigned char>("yei");

    if(colors3)
    {
        std::cout << "  colors found: " << colors3->numElements() << std::endl;
    }

}

void manipulatorUsage()
{
    size_t num_points = 10000;

    Channel<float> points(num_points, 3);
    Channel<float> normals(num_points, 3);
    Channel<unsigned char> colors(num_points, 3);
    
    fillChannel(points, 0.0f);
    fillChannel(normals, 1.0f);
    fillChannel(colors, static_cast<unsigned char>(255));

    std::cout << "3) Channel Manager Usage" << std::endl;

    // MultiChannelMap with extended functions -> ChannelManager

    BaseBuffer cm = {
        {"points2" , points}
    };
    
    cm["points"] = points;
    cm["colors"] = colors;
    cm["normals"] = normals;


    BaseBuffer cm_sliced = cm.manipulate(manipulators::Slice(10, 100));
    std::cout << "Sliced:" << std::endl;
    std::cout << cm_sliced << std::endl;

    BaseBuffer cm_sampled = cm.manipulate(manipulators::RandomSample(1000));
    std::cout << "Random Sampled:" << std::endl;
    std::cout << cm_sampled << std::endl;

    BaseBuffer cm_sliced_shallow = cm.manipulate(manipulators::SliceShallow(10, 100));

    // test shallow
    Channel<float> pts3 = cm_sliced_shallow.get<float>("points");
    pts3[0][0] = 42.0;
    Channel<float> pts4 = cm.get<float>("points");
    std::cout << "  should be 42: " << pts4[10][0] << std::endl;



    // clone
    BaseBuffer cm2 = cm.clone();

    Channel<float> pts = cm.get<float>("points");
    Channel<float> pts2 = cm2.get<float>("points");
    
    for(int i=0; i<pts.numElements(); i++)
    {
        for(int j=0; j<pts.width(); j++)
        {
            if(pts[i][j] != pts2[i][j])
            {
                std::cout << "ERROR" << std::endl;
            }
        }
    }
}

void compileTimeFunctionsUsage()
{
    size_t num_points = 10000;

    Channel<float> points(num_points, 3);
    Channel<float> normals(num_points, 3);
    Channel<unsigned char> colors(num_points, 3);

    BaseBuffer bb = {
        {"points", points},
        {"normals", normals},
        {"color", colors}
    };

    std::cout << "  index of float: " << BaseBuffer::index_of_type<float>::value << std::endl; 

    auto it = bb.find("points");
    if(it != bb.end())
    {
        BaseBuffer::val_type vchannel;
        vchannel = it->second;
        switch(vchannel.type())
        {
            case BaseBuffer::index_of_type<char>::value:
                std::cout << "  a char!" << std::endl;
                break;
            case BaseBuffer::index_of_type<float>::value:
                std::cout << "  a float! " << vchannel.numElements() << std::endl;
                break;
        }
    }

}

int main(int argc, const char** argv)
{
    basicChannelUsage();
    multiChannelMapUsage();
    channelManagerUsage();
    manipulatorUsage();
    compileTimeFunctionsUsage();

    return 0;
}