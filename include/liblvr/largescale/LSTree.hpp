

#ifndef LSTREE_HPP_
#define LSTREE_HPP_


// vorerst
#include <string>
#include <iostream>
#include <boost/iostreams/device/mapped_file.hpp>
#include "geometry/BoundingBox.hpp"
#include "geometry/Vertex.hpp"
#include "largescale/LSNode.hpp"

namespace lvr{
using namespace std;
/**
 * @brief A class for execution of a distribution using a very simple Kd-tree.
 * 	The classification criterion is to halve the longest axis.
 */
    class LSTree {
    public:

        LSTree(string mmfPath, int minP, int maxP, unsigned long long int mmf_size)
        {
            m_maxPoints=maxP;
            m_minPoints=minP;
            m_mmf.open(mmfPath);
            if(m_mmf.is_open())
            {
                m_data = (float *)m_mmf.data();
                Vertex<float> testm, tests;
                for(unsigned long long int i  = 0; i<mmf_size; i+=3)
                {
                    Vertex<float> tmp(m_data[i], m_data[i+1], m_data[i+2]);
                    if(i==0)
                    {
                        testm = Vertex<float>(tmp);
                        tests = Vertex<float>(tmp);
                    }
                    else{
                        if(testm<tmp) testm = Vertex<float>(tmp);
                        if(tmp<tests) tests = Vertex<float>(tmp);
                    }

                    //cout << tmp << endl;
                    m_bbox.expand(m_data[i], m_data[i+1], m_data[i+2]);
                    m_points.push_back(i);
                }
                std::cout << "MAXXXX: "<< testm << endl;
                std::cout <<"MINNNN" << tests << endl;
                std::cout << lvr::timestamp << "Files in vector: " << m_points.size()  << std::endl;
                std::cout << " Test Vertex: " << (m_data[m_points[0]]) << "|" << m_data[m_points[0]+1] << "|" << m_data[m_points[0]+2] << endl;
                std::cout << "Bounding Box: " << m_bbox << endl;
                LSNode * root = new LSNode();
                root->setPoints(m_points);
                root->setBoundingBox(m_bbox);
                splitPointCloud(root,0);

                int final_size=0;
                for(size_t i = 0; i< m_nodes.size() ; i++) {
                    final_size+=m_nodes.at(i)->getPoints().size();
                    cout << "node" << i << ": " << m_nodes.at(i)->getPoints().size() << endl;
                }
                cout << timestamp << "final size: " << final_size << " nodes: " << m_nodes.size() << endl;
            }
        }

    private:

        void splitPointCloud(LSNode * node, int again)
        {
            cout <<"-----"<< endl << timestamp << "size: " << node->getPoints().size() << " von " <<  m_maxPoints << endl;
            //Check if Node has enough points;
            if(node->getPoints().size()<m_maxPoints)
            {
                m_nodes.push_back(node);
            }
            else if(node->getPoints().size()==0  || node->again == 6)
            {
                node->getPoints().clear();
                delete node;
            }
            else
            {
                cout << timestamp << " Finding Splitingaxis" << endl;
                float splitAxisPoint, splitaxis;
                Vertex<float> diff = node->getBoundingBox().getMax() - node->getBoundingBox().getMin();

                float xlength = fabs(diff.x) ;
                float ylength = fabs(diff.y);
                float zlength = fabs(diff.z);
                Vertex<float> maxVertex = node->getBoundingBox().getMax();
                Vertex<float> minVertex = node->getBoundingBox().getMin();
                //Find longest Axis
                if ( xlength > ylength )
                {
                    //Split x-Axsis
                    if ( xlength > zlength)
                    {
                        splitaxis = 0;
                        if      (again == 0) splitAxisPoint = ( minVertex.x + maxVertex.x)/2.0f;
                        else if (again == 1) splitAxisPoint = ( ( minVertex.x + maxVertex.x ) / 2.0f ) + ( ( (node->again * 2) * ( maxVertex.x - minVertex.x)) / 11 );
                        else if (again == 2) splitAxisPoint = ( ( minVertex.x + maxVertex.x ) / 2.0f ) - ( ( (node->again * 2) * ( maxVertex.x - minVertex.x)) / 11 );
                    }
                    //Split z-Axsis
                    else
                    {
                        splitaxis = 2;
                        if      (again == 0) splitAxisPoint = ( minVertex.z + maxVertex.z)/2.0f;
                        else if (again == 1) splitAxisPoint = ( ( minVertex.z + maxVertex.z ) / 2.0f ) + ( ( (node->again * 2) * ( maxVertex.z - minVertex.z)) / 11 );
                        else if (again == 2) splitAxisPoint = ( ( minVertex.z + maxVertex.z ) / 2.0f ) - ( ( (node->again * 2) * ( maxVertex.z - minVertex.z)) / 11 );
                    }
                }
                else
                {
                    //Split y-Axsis
                    if ( ylength > zlength)
                    {
                        splitaxis = 1;
                        if      (again == 0) splitAxisPoint = ( minVertex.y + maxVertex.y)/2.0f;
                        else if (again == 1) splitAxisPoint = ( ( minVertex.y + maxVertex.y ) / 2.0f ) + ( ( (node->again * 2) * ( maxVertex.y - minVertex.y)) / 11 );
                        else if (again == 2) splitAxisPoint = ( ( minVertex.y + maxVertex.y ) / 2.0f ) - ( ( (node->again * 2) * ( maxVertex.y - minVertex.y)) / 11 );
                    }
                    //Split z-Axsis
                    else
                    {
                        splitaxis = 2;
                        if      (again == 0) splitAxisPoint = ( minVertex.z + maxVertex.z)/2.0f;
                        else if (again == 1) splitAxisPoint = ( ( minVertex.z + maxVertex.z ) / 2.0f ) + ( ( (node->again * 2) * ( maxVertex.z - minVertex.z)) / 11 );
                        else if (again == 2) splitAxisPoint = ( ( minVertex.z + maxVertex.z ) / 2.0f ) - ( ( (node->again * 2) * ( maxVertex.z - minVertex.z)) / 11 );
                    }
                }

                // count the loops with the same pointcloud
                if ( again != 0) node->again++;
                if (minVertex == maxVertex)
                {
                    cout << timestamp << "something went wrong (minVertex == maxVertex)" << endl;
                    return;
                }

                cout << timestamp << "splitting axis: " << splitaxis << " at " << splitAxisPoint << endl;

                LSNode * child1 = new LSNode();
                LSNode * child2 = new LSNode();

                Vertex<float> child2_min = minVertex;
                Vertex<float> child1_max = maxVertex;
                child2_min[splitaxis] = splitAxisPoint;
                child1_max[splitaxis] = splitAxisPoint;
                BoundingBox<Vertex<float>> bbchild1(minVertex, child1_max);
                BoundingBox<Vertex<float>> bbchild2(child2_min, maxVertex);
                cout << timestamp << "setting bounding boxes of children" << endl << bbchild1 <<endl<< bbchild2 << endl;

                child1->setBoundingBox(bbchild1);
                child2->setBoundingBox(bbchild2);
                cout << timestamp << "checking witch points belong to witch child" << endl;
                vector<unsigned long long int> c1_points;
                vector<unsigned long long int> c2_points;
                for(size_t i = 0; i< node->getPoints().size(); i++)
                {

                    Vertex<float> point(m_data[node->getPoints().at(i)], m_data[node->getPoints().at(i)+1], m_data[node->getPoints().at(i)+2]);

                    if(point[splitaxis]<=splitAxisPoint)
                    {
                        child1->getPoints().push_back(node->getPoints().at(i));
                        //cout << point<< endl << "in 1" << endl;
                    }
                    else
                    {
                        //cout << point<< endl << "in 2" << endl;
                        child2->getPoints().push_back(node->getPoints().at(i));
                    }

                }
                cout << timestamp << "CSizes: " << child1->getPoints().size() << "|"<< child2->getPoints().size() << endl;
                if((child1->getPoints().size() < m_minPoints) || (child2->getPoints().size() < m_minPoints))
                {
                    cout << timestamp<< "one child smaler then minPoints, again=" << node->again<< endl;
                    if ( child1->getPoints().size() == 0 )
                    {
                        cout << timestamp<< "child1 = 0"<< endl;
                        child1->getPoints().clear();
                        delete child1;
                        splitPointCloud(child2, 0);
                    }
                    else if (child2->getPoints().size() == 0 )
                    {
                        cout << timestamp<< "child1 = 2"<< endl;
                        child2->getPoints().clear();
                        delete child2;
                        splitPointCloud(child1, 0);
                    }
                    else
                    {

                        if ( child1->getPoints().size() < child2->getPoints().size())
                        {
                            delete child1;
                            delete  child2;
                            splitPointCloud(node, 1);
                        }
                        else
                        {
                            delete child1;
                            delete  child2;
                            splitPointCloud(node, 2);
                        }
                    }
                }
                else
                {
                    cout << "child size: " << child1->getPoints().size() << "|" << child2->getPoints().size() << endl;
                    node->getPoints().clear();

                    splitPointCloud(child1, 0);
                    splitPointCloud(child2, 0);
                    delete node;
                }



                cout << timestamp << " Child1+2: " << child1->getPoints().size() + child2->getPoints().size() << endl;


            }
        }



        vector<LSNode*> m_nodes;
        vector<unsigned long long int> m_points;
        boost::iostreams::mapped_file_source m_mmf;
        BoundingBox<Vertex<float>> m_bbox;
        int m_maxPoints;
        int m_minPoints;
        float * m_data;
    };
}


#endif /* LSTREE_HPP_ */
