namespace lvr2
{

template<typename T, int Rows, int Cols>
Eigen::Matrix<T, Rows, Cols> loadMatrixFromYAML(const YAML::const_iterator& it)
{
    // Alloc memory for matrix entries
    T data[Rows * Cols] = {0};

    // Read entries
    int c = 0;
    for (auto& i : it->second)
    {
        if(c < Rows * Cols)
        {
            data[c++] = i.as<T>();
        }
        else
        {
            std::cout << timestamp << "Warning: Load Matrix from YAML: Buffer overflow." << std::endl;
            break;
        }
    }
    return Eigen::Map<Eigen::Matrix<T, Rows, Cols>>(data);
}

template<typename T, int Rows, int Cols>
void saveMatrixToYAML(YAML::Node& node, const std::string& name, const Eigen::Matrix<T, Rows, Cols>& matrix)
{
    const T* data = matrix.data();
    for(size_t i = 0; i < Rows * Cols; i++)
    {
        
        node[name].push_back(data[i]);
    }
}

template<typename T>
void loadArrayFromYAML(const YAML::const_iterator& it, T* array, size_t n)
{
    // Read entries
    int c = 0;
    for (auto &i : it->second)
    {
        if (c < n)
        {
            array[c++] = i.as<T>();
        }
        else
        {
            std::cout << timestamp << "Warning: Load Array from YAML: Buffer overflow." << std::endl;
            break;
        }
    }
}

} // namespace lvr2