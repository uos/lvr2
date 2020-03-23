namespace lvr2
{

template <typename Derived>
void ScanImageIO<Derived>::save(
    const std::string& group, 
    const std::string& container, 
    const ScanImagePtr& buffer)
{
    // TODO
}



template <typename Derived>
ScanImagePtr ScanImageIO<Derived>::load(
    const size_t& scanPos, 
    const size_t& camNr, 
    const size_t& imgNr)
{
    ScanImagePtr ret;

    // TODO

    return ret;
}

template <typename Derived>
ScanImagePtr ScanImageIO<Derived>::load(
    const std::string& group, 
    const std::string& container)
{
    ScanImagePtr ret;

    // check wether the given group is type ScanProjectIO

    // TODO

    return ret;
}

template <typename Derived>
bool ScanImageIO<Derived>::isScanImage(HighFive::Group& group)
{
    return true;
}

} // namespace lvr2
