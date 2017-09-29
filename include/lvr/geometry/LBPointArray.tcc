
namespace lvr {

// static helper methods

template<typename T>
static void mallocPointArray(LBPointArray<T>& m) {

    m.elements = (T*)malloc(m.width * m.dim * sizeof(T));

}

template<typename T>
static void generatePointArray(LBPointArray<T>& m, int width, int dim)
{

    m.dim = dim;
    m.width = width;
    m.elements = (T*)malloc(m.width * m.dim * sizeof(T) );

}

template<typename T>
static void generatePointArray(int id, LBPointArray<T>& m, int width, int dim)
{

    m.dim = dim;
    m.width = width;
    m.elements = (T*)malloc(m.width * m.dim * sizeof(T) );

}

template<typename T>
static void fillPointArrayWithSequence(LBPointArray<T>& m) {

    for(unsigned int i=0;i<m.width*m.dim;i++)
    {
        m.elements[i] = i;
    }

}

// Pool function
template<typename T>
static void fillPointArrayWithSequence(int id, LBPointArray<T>& m) {

    for(unsigned int i=0; i<m.width*m.dim; i++)
    {
        m.elements[i] = i;
    }

}

template<typename T>
static void copyVectorInterval(LBPointArray<T>& in, int start, int end, LBPointArray<T>& out) {

    for(int i=0; i < (end-start); i++)
    {
        out.elements[i] = in.elements[i + start];
    }
}

template<typename T>
static void copyDimensionToPointArray(LBPointArray<T>& in, int dim, LBPointArray<T>& out) {

    for(int i = 0; i<out.width; i++)
    {
        out.elements[i] = in.elements[i * in.dim + dim];
    }
}

template<typename T>
static void splitPointArray(LBPointArray<T>& I, LBPointArray<T>& I_L, LBPointArray<T>& I_R) {
    
    unsigned int i=0;
    for(; i < I_L.width * I_L.dim; i++){
        I_L.elements[i] = I.elements[i];
    }
    unsigned int j=0;
    for(; i<I.width*I.dim && j<I_R.width*I_R.dim; i++, j++){
        I_R.elements[j] = I.elements[i];
    }

    

}

template<typename T, typename U>
static bool checkSortedIndices(const LBPointArray<T>& V, const LBPointArray<U>& sorted_indices, unsigned int dim, int n)
{
    bool check = true;
    volatile U last_index = sorted_indices.elements[0];
    if(last_index > V.width)
    {
        std::cout << n <<" BLAAA falsch " << std::endl;
    }
    for(U i=1; i<sorted_indices.width; i++)
    {
        volatile U index = sorted_indices.elements[i];
        if(index > V.width )
        {
            std::cout << n << " index: "<< index << " to high: max_size "<< V.width << std::endl;
            std::cout << n << " cursed by: " << sorted_indices.elements[i] << std::endl;
            check = false;
            //exit (EXIT_FAILURE);
            continue;
        }

        if(last_index > V.width)
        {
            std::cout << n << " last index: "<< last_index << " to high max_size "<< V.width << std::endl;
            std::cout << n << " cursed by: " << sorted_indices.elements[i] << std::endl;
            check = false;
            //exit (EXIT_FAILURE);
            continue;
        }

        if( V.elements[V.dim * index + dim ] < V.elements[V.dim * last_index + dim ] )
        {
            // std::cout << " Index " << index << " Wrong" << std::endl;
            // std::cout << V.elements[V.dim * index + dim] << " < " << V.elements[V.dim * last_index + dim ] << std::endl;
            check = false;
        }
        last_index = index;
    }
    return check;
}

template<typename T, typename U>
static void splitPointArrayWithValue(const LBPointArray<T>& V, const LBPointArray<U>& I, LBPointArray<U>& I_L, LBPointArray<U>& I_R, int current_dim, T value,
                T& deviation_left, T& deviation_right, const unsigned int& orig_dim,
                const std::list<U>& critical_indices_left_copy, const std::list<U>& critical_indices_right_copy)
{

    std::list<U> critical_indices_left(critical_indices_left_copy);
    std::list<U> critical_indices_right(critical_indices_right_copy);

    U i_l = 0;
    U i_r = 0;

    T smallest_left = std::numeric_limits<T>::max();
    T biggest_left = std::numeric_limits<T>::lowest();
    
    T biggest_right = std::numeric_limits<T>::lowest();
    T smallest_right = std::numeric_limits<T>::max();

    U counter_loop = 0;

    for(int i=0; i<I.width; i++)
    {

        T current_value = V.elements[ I.elements[i] * V.dim + current_dim ];
        T dim_value = V.elements[ I.elements[i] * V.dim + orig_dim ];
        //~ printf("curr val: %f\n", current_value);
        if(current_value < value && I_L.width > i_l ){
            if(dim_value < smallest_left )
            {
                smallest_left = dim_value;
            }
            if(dim_value > biggest_left)
            {
                biggest_left = dim_value;
            }
            //~ printf("add to left: %f with value %f\n", I.elements[i], current_value);
            I_L.elements[i_l] = I.elements[i];
            i_l++;
        } else if(current_value > value && I_R.width > i_r){
            //~ printf("add to right: %f with value %f\n", I.elements[i], current_value);
            if(dim_value > biggest_right)
            {
                biggest_right = dim_value;
            }
            if(dim_value < smallest_right)
            {
                smallest_right = dim_value;
            }
            I_R.elements[i_r] = I.elements[i];
            i_r++;
        } else {
            
            bool found = false;

            for(auto it = critical_indices_left.begin();
                it != critical_indices_left.end();
                it++)
            {
                // std::cout << *it << std::endl;
                if(*it == I.elements[i] )
                {
                    I_L.elements[i_l] = I.elements[i];
                    i_l++;
                    found = true;
                    critical_indices_left.erase(it);
                    break;
                }
            }

            if(!found)
            {
                for(auto it = critical_indices_right.begin();
                    it != critical_indices_right.end();
                    it++)
                {
                    // std::cout << *it << std::endl;
                    if(*it == I.elements[i] )
                    {
                        I_R.elements[i_r] = I.elements[i];
                        i_r++;
                        found = true;
                        critical_indices_right.erase(it);
                        break;
                    }
                }
            }

            if(!found)
            {
                std::cout << "KAPUTT" << std::endl;
                std::cout << "cant find " << I.elements[i] << std::endl;
                
                exit (EXIT_FAILURE);
            }
            
        }

    }


    deviation_left = biggest_left - smallest_left;
    deviation_right = biggest_right - smallest_right;
}


template<typename T, typename U>
static void splitPointArrayWithValueSet(const LBPointArray<T>& V, const LBPointArray<U>& I, LBPointArray<U>& I_L, LBPointArray<U>& I_R, int current_dim, T value,
                T& deviation_left, T& deviation_right, const unsigned int& orig_dim,
                std::unordered_set<U>& critical_indices_left, std::unordered_set<U> critical_indices_right)
{

    U i_l = 0;
    U i_r = 0;

    T smallest_left = std::numeric_limits<T>::max();
    T biggest_left = std::numeric_limits<T>::lowest();
    
    T biggest_right = std::numeric_limits<T>::lowest();
    T smallest_right = std::numeric_limits<T>::max();

    U counter_loop = 0;

    // U old_index = 1;

    for(int i=0; i<I.width; i++)
    {
        // std::cout << I.elements[i] << std::endl;


        // if(old_index == 0 && I.elements[i] == 0 )
        // {
        //     exit(1);
        // }

        // old_index = I.elements[i];

        T current_value = V.elements[ I.elements[i] * V.dim + current_dim ];
        T dim_value = V.elements[ I.elements[i] * V.dim + orig_dim ];
        //~ printf("curr val: %f\n", current_value);
        if(current_value < value && I_L.width > i_l ){
            if(dim_value < smallest_left )
            {
                smallest_left = dim_value;
            }
            if(dim_value > biggest_left)
            {
                biggest_left = dim_value;
            }
            //~ printf("add to left: %f with value %f\n", I.elements[i], current_value);
            I_L.elements[i_l] = I.elements[i];
            i_l++;
        } else if(current_value > value && I_R.width > i_r){
            //~ printf("add to right: %f with value %f\n", I.elements[i], current_value);
            if(dim_value > biggest_right)
            {
                biggest_right = dim_value;
            }
            if(dim_value < smallest_right)
            {
                smallest_right = dim_value;
            }
            I_R.elements[i_r] = I.elements[i];
            i_r++;
        } else {
            
            bool found = false;

            auto critical_it = critical_indices_left.find( I.elements[i] );

            if(critical_it != critical_indices_left.end())
            {
                I_L.elements[i_l] = I.elements[i];
                i_l++;
                found = true;
                critical_indices_left.erase(critical_it);
            }

            if(!found)
            {
                critical_it =  critical_indices_right.find(I.elements[i]);
                if(critical_it != critical_indices_right.end())
                {
                    I_R.elements[i_r] = I.elements[i];
                    i_r++;
                    found = true;
                    critical_indices_right.erase(critical_it);
                }
            }

            if(!found)
            {
                // std::cout << "KAPUTT" << std::endl;
                // std::cout << "cant find " << i << " " << I.elements[i] << std::endl;
                if(I_L.width > i_l)
                {
                    I_L.elements[i_l] = I.elements[i];
                    i_l++;
                }else{
                    I_R.elements[i_r] = I.elements[i];
                    i_r++;
                }
            }
            
        }

    }


    deviation_left = biggest_left - smallest_left;
    deviation_right = biggest_right - smallest_right;
}

template<typename T>
static unsigned int checkNumberOfBiggerValues(LBPointArray<T>& V, unsigned int dim, T split)
{
    unsigned int result = 0;
    for(unsigned int i=0; i<V.width; i++)
    {
        if(V.elements[ i*V.dim + dim] > split)
        {
            result++;
        }
    }
    return result;
}

static unsigned int checkNumberOfSmallerEqualValues(LBPointArray<float>& V, unsigned int dim, float split)
{
    unsigned int result = 0;
    for(unsigned int i=0; i<V.width; i++)
    {
        if(V.elements[ i*V.dim + dim] <= split)
        {
            result++;
        }
    }
    return result;
}


// SORT FUNCTIONS THREADED
template<typename T, typename U>
static void mergeHostWithIndices(T* a, U* b, unsigned int i1, unsigned int j1, unsigned int i2, unsigned int j2, int limit) {

    int limit_end = limit;

    T* temp = (T*) malloc((j2-i1+1) * sizeof(T));  //array used for merging
    U* temp_indices = (U*) malloc((j2-i1+1) * sizeof(U));  //array used for merging

    unsigned int i,j,k;
    i=i1;    //beginning of the first list
    j=i2;    //beginning of the second list
    k=0;

    unsigned int counter = 0;

    while( i<=j1 && j<=j2 && limit!=0 )    //while elements in both lists
    {
        counter ++;
        limit--;
        if(a[i]<a[j]){
            temp_indices[k] = b[i];
            temp[k++] = a[i++];

        }else{
            temp_indices[k] = b[j];
            temp[k++]=a[j++];
        }
    }

    while(i <= j1 && limit != 0) //copy remaining elements of the first list
    {
        temp_indices[k] = b[i];
        temp[k++]=a[i++];
    }

    while(j <= j2 && limit!=0 ) {   //copy remaining elements of the second list
        temp_indices[k] = b[j];
        temp[k++]=a[j++];
    }

    //Transfer elements from temp[] back to a[]
    for(i=i1,j=0;i<=j2 && limit_end!=0 ;i++,j++,limit_end--)
    {
        b[i] = temp_indices[j];
        a[i] = temp[j];
    }

    free(temp_indices);
    free(temp);
}

template<typename T, typename U>
static void naturalMergeSort(LBPointArray<T>& in, int dim, LBPointArray<U>& indices, LBPointArray<T>& m, int limit) {

    copyDimensionToPointArray<T>(in, dim, m);

    unsigned int m_elements = m.width * m.dim;

    unsigned int slide_buffer_size = static_cast<unsigned int>(m_elements-0.5);
    U* slide_buffer = (U*) malloc(slide_buffer_size * sizeof(U));


    //create RUNS
    unsigned int num_slides = 1;
    slide_buffer[0] = 0;
    for(unsigned int i=1; i < slide_buffer_size+1; i++)
    {
        if(m.elements[i] < m.elements[i-1])
        {
            slide_buffer[num_slides] = i;
            num_slides++;
        }

    }
    slide_buffer[num_slides] = m_elements;
    slide_buffer_size = num_slides+1;


    //sort
    unsigned int count = 0;
    int current_limit = -1;
    while(num_slides > 1)
    {
        if(num_slides > 2){
            current_limit = limit;
        }

        unsigned int i;
        for(i=2;i<num_slides+1;i+=2)
        {
            mergeHostWithIndices<T,U>(m.elements, indices.elements , slide_buffer[i-2], slide_buffer[i-1]-1, slide_buffer[i-1], slide_buffer[i]-1, current_limit);

            slide_buffer[i/2] = slide_buffer[i];
        }

        if(num_slides%2 == 1){
            slide_buffer[(num_slides+1)/2] = slide_buffer[num_slides];
        }

        count ++;
        num_slides = static_cast<unsigned int>(num_slides/2.0+0.5);

    }

    free(slide_buffer);
}

template<typename T, typename U>
static void sortByDim(LBPointArray<T>& V, int dim, LBPointArray<U>& indices, LBPointArray<T>& values) {

    naturalMergeSort<T,U>(V, dim, indices, values);

}

template<typename T, typename U>
static void generateAndSort(int id, LBPointArray<T>& vertices, LBPointArray<U>* indices_sorted, LBPointArray<T>* values_sorted, int dim)
{
    generatePointArray<U>( indices_sorted[dim], vertices.width, 1);
    generatePointArray<T>( values_sorted[dim], vertices.width, 1);
    
    fillPointArrayWithSequence<U>( indices_sorted[dim] );

    sortByDim<T,U>( vertices, dim, indices_sorted[dim] , values_sorted[dim] );

}

} /* namespace lvr */