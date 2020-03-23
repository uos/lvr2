namespace lvr2
{

template <typename Derived>
void ScanCameraIO<Derived>::save(
    const std::string& group, 
    const std::string& container, ScanCameraPtr& buffer)
{
    // TODO
}

template <typename Derived>
ScanCameraPtr ScanCameraIO<Derived>::load(
    const std::string& group, 
    const std::string& container)
{
    ScanCameraPtr ret;

    // TODO

    return ret;
}



template <typename Derived>
bool ScanCameraIO<Derived>::isScanCamera(const std::string& group)
{
    return true;
}

} // namespace lvr2
