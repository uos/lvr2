#include <iostream>

#include "lvr2/types/Channel.hpp"
#include "lvr2/types/VariantChannel.hpp"
#include "lvr2/types/VariantChannelMap.hpp"
#include "lvr2/types/MultiChannelMap.hpp"

#include "lvr2/types/ChannelManager.hpp"


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

void channelTest()
{
    std::cout << "Channel Test" << std::endl;

    size_t num_points = 10000;
    Channel<float> points(num_points, 3);
    fillChannel(points, 0.0f);
    Channel<float> normals(num_points, 3);
    fillChannel(normals, 1.0f);
    Channel<unsigned char> colors(num_points, 3);
    fillChannel(colors, static_cast<unsigned char>(255));


    // VariantChannel<float, int> achannel;
    // VariantChannel<float, unsigned char> vchannel = normals;

    VariantChannel<float, unsigned char> vchannel(Channel<float>(num_points, 3));
    

    std::cout << vchannel << std::endl;

    std::cout << vchannel.numElements() << std::endl;

    auto data = vchannel.dataPtr<float>();

    std::cout << VariantChannel<float, unsigned char>::index_of_type<float>::value << std::endl;

    std::cout << data[0] << std::endl;

    MultiChannelMap cm;
    cm["points"] = points;
    cm["colors"] = colors;
    cm.insert({"normals", normals});

    std::cout << cm.size() << std::endl;

    std::cout << "BLAASAA" << std::endl;

    std::cout << cm["points"].width() << std::endl;

    for(auto it: cm)
    {
        std::cout << it.first << ": " << it.second.numElements() << " x " << it.second.width() << std::endl;
    
        if(it.second.is_type<float>())
        {
            std::cout << "float !! " << std::endl;

            
            cm.get<float>("points");
            Channel<float> data = cm.get<float>(it.first);
        }
    }

    std::cout << cm << std::endl;
}

void channelManagerTest()
{
    std::cout << "ChannelManager Test" << std::endl;

    size_t num_points = 10000;
    Channel<float> points(num_points, 3);
    fillChannel(points, 0.0f);
    Channel<float> normals(num_points, 3);
    fillChannel(normals, 1.0f);
    Channel<unsigned char> colors(num_points, 3);
    fillChannel(colors, static_cast<unsigned char>(255));

    // initializer list
    ChannelManager cm = {
        {"points2" , points}
    };
    
    cm["points"] = points;
    cm["colors"] = colors;
    cm["normals"] = normals;

    FloatChannelOptional points_test;

    points_test = cm.getFloatChannel("points");
    if(!points_test)
    {
        std::cerr << "doof" << std::endl;
    } else {
        std::cout << "points: " << points_test->numElements() << std::endl;
    }

    points_test = cm.getFloatChannel("colors");
    if(points_test)
    {
        std::cerr << "doof" << std::endl;
    } else {
        std::cout << "success" << std::endl;
    }

    points_test = cm.getFloatChannel("asdf");
    if(points_test)
    {
        std::cerr << "doof" << std::endl;
    } else {
        std::cout << "success" << std::endl;
    }

    cm.getChannel("points", points_test);
    if(!points_test)
    {
        std::cerr << "doof" << std::endl;
    } else {
        std::cout << "points: " << points_test->numElements() << std::endl;
    }

    cm.getChannel("colors", points_test);
    if(points_test)
    {
        std::cerr << "doof" << std::endl;
    } else {
        std::cout << "success" << std::endl;
    }

    cm.getChannel("asdf", points_test);
    if(points_test)
    {
        std::cerr << "doof" << std::endl;
    } else {
        std::cout << "success" << std::endl;
    }


    cm.addFloatAtomic(5.5, "myatomic");
    auto myatomic = cm.getFloatAtomic("myatomic");
    if(myatomic)
    {
        std::cout << "myatomic is " << *myatomic << std::endl;
    }

    myatomic = cm.getFloatAtomic("bla");
    if(!myatomic)
    {
        std::cout << "success" << std::endl;
    }

    std::cout << "total number of elements: " << cm.size() << std::endl;
    std::cout << "num float channels: " << cm.numChannels<float>() << std::endl;
    std::cout << cm << std::endl;

}

int main(int argc, const char** argv)
{
    // channelTest();
    channelManagerTest();

    return 0;
}