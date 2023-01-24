#include "lvr2/types/DistortionModels.hpp"

namespace lvr2
{

DistortionModel::DistortionModel()
: DistortionModel(UnknownDistortion())
{}

DistortionModel::DistortionModel(const DistortionModel& other)
: impl_(other.impl_->clone())
{}

DistortionModel::DistortionModel(DistortionModel&& other)
: impl_(std::move(other.impl_))
{}

DistortionModel& DistortionModel::operator=(const DistortionModel& other)
{
    this->impl_ = std::move(other.impl_->clone());
    return *this;
}

DistortionModel& DistortionModel::operator=(DistortionModel&& other)
{
    this->impl_ = std::move(other.impl_);
    return *this;
}

UnknownDistortion::UnknownDistortion(const std::vector<double>& coeffs)
: coefficients_(coeffs)
{}

RieglDistortion::RieglDistortion(const std::vector<double>& coeffs)
: coefficients_(coeffs)
{}

OpenCVDistortion::OpenCVDistortion(const std::vector<double>& coeffs)
: coefficients_(coeffs)
{}

DistortionModel DistortionModel::fromName(const std::string& name, const std::vector<double>& coefficients)
{
    if (name == OpenCVDistortion::type)
    {
        return OpenCVDistortion{coefficients};
    }
    else if (name == RieglDistortion::type)
    {
        return RieglDistortion{coefficients};
    }
    else if (name == UnknownDistortion::type)
    {
        return UnknownDistortion{coefficients};
    }

    throw std::invalid_argument("[DistortionModel::fromName] Unimplemented distortion model '" + name + "'");
}


} // namespace lvr2