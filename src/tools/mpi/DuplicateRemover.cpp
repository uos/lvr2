#include "DuplicateRemover.hpp"

using namespace std;
using namespace lvr;
namespace lvr
{

MeshBufferPtr DuplicateRemover::removeDuplicates(MeshBufferPtr mBuffer)
{

    lvr::timestamp.resetTimer();

    size_t vAmount, fAmount;
    floatArr vArray = mBuffer->getVertexArray(vAmount);
    uintArr  fArray = mBuffer->getFaceArray(fAmount);
    std::vector<unsigned int> faceArray(fArray.get(), fArray.get() + (fAmount*3));

    //vArray.reset();
    fArray.reset();

    cout << lvr::timestamp << "god model" << endl;
    cout << lvr::timestamp << "inserting vertices to set" << endl;
    vector<sortPoint<float> > vertexSet;
    vertexSet.reserve(vAmount);
    for(size_t i = 0 ; i<vAmount; i++)
    {
       vertexSet.push_back(sortPoint<float>(&vArray.get()[i*3], i));

    }
    vertexSet.shrink_to_fit();
    std::sort(vertexSet.begin(), vertexSet.end());
    auto vend = std::unique(vertexSet.begin(), vertexSet.end());
    size_t old_vsize = vertexSet.size();
    vertexSet.resize( std::distance(vertexSet.begin(),vend) );
    size_t new_vsize = vertexSet.size();


    cout << lvr::timestamp << "mapping old to new indices" << endl;
    std::unordered_map<unsigned int, unsigned int> oldnew;
    oldnew.reserve(vAmount);

    size_t vertexamount = vAmount;

    int pro = 0;

    lvr::ProgressBar pg1(vertexamount, "mapping old to new indices..." );
    for(size_t i = 0 ; i<vertexamount; ++i)
    {

            ++pg1;



        sortPoint<float> tmp_point(&vArray.get()[i*3], i);

        // Todo: search only in removed vertices (make modified unique fuction)
       auto it = std::lower_bound(vertexSet.begin(), vertexSet.end(),tmp_point);
       if (*(it) == tmp_point)
       {
           size_t dist = std::distance(vertexSet.begin(), it);
           oldnew[i]= dist;
       }



    }
    cout << "checking face indices..." << endl;
    pro = 0;
    unsigned int swapped = 0;
    int fi = 0;
     lvr::ProgressBar pg2(faceArray.size(), "checking face indices..." );
    for(auto it = faceArray.begin(), end = faceArray.end(); it != end; ++it)
    {

        ++pg2;

        if(oldnew.find(*it)!=oldnew.end())
        {
            *(it) = oldnew[*(it)];
        }
        // if face point not found, delete that face (should not happen)
        else
        {
            cout << "wtf" << endl;
            unsigned int dist = std::distance(faceArray.begin(), it) / 3;
            dist*=3;
            std::iter_swap(faceArray.begin() + dist, faceArray.end()-3-swapped);
            std::iter_swap(faceArray.begin() + dist +1, faceArray.end()-2-swapped);
            std::iter_swap(faceArray.begin() + dist +2, faceArray.end()-1-swapped);
            swapped+=3;
            it = faceArray.begin() + dist -1;
        }
        fi++;
    }
    faceArray.resize(std::distance(faceArray.begin(),faceArray.end() - swapped ));

    cout << lvr::timestamp << "copying faces" << endl;
    vector<sortPoint<unsigned int> > newFaces;
    newFaces.reserve(faceArray.size()/3);
    for(int i = 0 ; i<faceArray.size() ; i+=3)
    {
        newFaces.push_back(sortPoint<unsigned int>(&faceArray[i]));
    }
    cout << lvr::timestamp << "sorting faces" << endl;

    std::sort(newFaces.begin(), newFaces.end());
    cout << lvr::timestamp << "get duplicate faces" << endl;
    auto fend = std::unique(newFaces.begin(), newFaces.end());
     size_t old_fsize = newFaces.size();
    cout << lvr::timestamp << "remove duplicate faces" << endl;

    newFaces.resize( std::distance(newFaces.begin(),fend) );
    size_t new_fsize = newFaces.size();


    cout << lvr::timestamp << "making new varray" << endl;

    std::vector<float> newVertexArray;
    newVertexArray.reserve(vertexSet.size()*3);
    for(auto it = vertexSet.begin(), end = vertexSet.end(); it != end; ++it)
    {
        newVertexArray.push_back(it->x());
        newVertexArray.push_back(it->y());
        newVertexArray.push_back(it->z());
    }
    cout << lvr::timestamp << "making new farray" << endl;

    std::vector<unsigned int> newFaceArray;
    newFaceArray.reserve(newFaces.size()*3);
    for(auto it = newFaces.begin(), end = newFaces.end(); it != end; ++it)
    {
        newFaceArray.push_back(it->x());
        newFaceArray.push_back(it->y());
        newFaceArray.push_back(it->z());
    }


    cout << lvr::timestamp << "saving model" << endl;
    MeshBufferPtr completeMesh(new MeshBuffer);
    completeMesh->setFaceArray(newFaceArray);
    completeMesh->setVertexArray(newVertexArray);

    cout << lvr::timestamp << "finished, removed " <<old_vsize-new_vsize << " vertices and " << old_fsize - new_fsize << " faces" << endl;

    return completeMesh;
}
}