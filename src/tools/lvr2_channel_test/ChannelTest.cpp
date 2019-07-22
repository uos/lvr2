#include <iostream>

#include "lvr2/types/Channel.hpp"
#include "lvr2/types/VariantChannel.hpp"
#include "lvr2/types/VariantChannelMap.hpp"
#include "lvr2/types/MultiChannelMap.hpp"


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

// void pointCloudTest()
// {
//     std::cout << "PointCloud Test" << std::endl;

//     size_t num_points = 10000;
//     Channel<float> points(num_points, 3);
//     fillChannel(points, 0.0f);
//     Channel<float> normals(num_points, 3);
//     fillChannel(normals, 1.0f);
//     Channel<unsigned char> colors(num_points, 3);
//     fillChannel(colors, static_cast<unsigned char>(255));

//     PointCloud pc;
//     pc["points"] = points;

//     std::cout << pc << std::endl;

// }

int main(int argc, const char** argv)
{
    channelTest();
    // pointCloudTest();

    return 0;
}