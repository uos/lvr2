#pragma once
#include "DistortionModels.hpp"

namespace lvr2
{

template <typename ModelT>
DistortionModel::DistortionModel(const ModelT& model)
: impl_(std::make_unique<Model<ModelT>>(model))
{}

template <typename ModelT>
DistortionModel::DistortionModel(std::remove_reference_t<ModelT>&& model)
: impl_(std::make_unique<Model<ModelT>>(std::move(model)))
{}

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

Vector2d DistortionModel::distortPoint(const Vector2d& p) const
{
    return impl_->distortPoint(p);
}

const std::vector<double>& DistortionModel::coefficients() const
{
    return impl_->coefficients();
}

std::string DistortionModel::name() const
{
    return impl_->name();
}

template <typename Impl>
DistortionModel::Model<Impl>::Model(const std::remove_reference_t<Impl>& obj)
: obj_(obj)
{}

template <typename Impl>
DistortionModel::Model<Impl>::Model(std::remove_reference_t<Impl>&& obj)
: obj_(std::move(obj))
{}

template <typename Impl>
std::unique_ptr<DistortionModel::Concept> DistortionModel::Model<Impl>::clone() const
{
    return std::make_unique<Model<Impl>>(obj_);
}

template <typename Impl>
Vector2d DistortionModel::Model<Impl>::distortPoint(const Vector2d& p) const
{
    return obj_.distortPoint(p);
}
template <typename Impl>
const std::vector<double>& DistortionModel::Model<Impl>::coefficients() const
{
    return obj_.coefficients();
}
template <typename Impl>
std::string DistortionModel::Model<Impl>::name() const
{
    return obj_.name();
}

RieglDistortion::RieglDistortion(const std::vector<double>& coeffs)
: coefficients_(coeffs)
{}

Vector2d RieglDistortion::distortPoint(const Vector2d& p) const
{
    const double& k1 = coefficients_[0];
    const double& k2 = coefficients_[1];
    const double& p1 = coefficients_[2];
    const double& p2 = coefficients_[3];
    const double& k3 = coefficients_[4];
    const double& k4 = coefficients_[5];

    const double x = p.x();
    const double y = p.y();

    const double r_2 = std::pow(std::atan(std::sqrt(std::pow(x, 2) + std::pow(y, 2))), 2);
    const double r_4 = std::pow(r_2, 2);
    const double r_6 = std::pow(r_2, 3);
    const double r_8 = std::pow(r_2, 4);

    const double ud = x*(1 + k1*r_2 + k2*r_4 + k3*r_6 + k4*r_8) + 2*x*y*p1 + p2*(r_2 + 2*std::pow(x, 2));
    const double vd = y*(1 + k1*r_2 + k2*r_4 + k3*r_6 + k4*r_8) + 2*x*y*p2 + p1*(r_2 + 2*std::pow(y, 2));

    return lvr2::Vector2d(ud, vd);
}

std::string RieglDistortion::name() const
{
    return type;
}

const std::vector<double>& RieglDistortion::coefficients() const
{
    return coefficients_;
}

OpenCVDistortion::OpenCVDistortion(const std::vector<double>& coeffs)
: coefficients_(coeffs)
{}

Vector2d OpenCVDistortion::distortPoint(const Vector2d& p) const
{
    // https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
    // Distort points for the default OpenCV Model
    const double& k1 = coefficients_[0];
    const double& k2 = coefficients_[1];
    const double& p1 = coefficients_[2];
    const double& p2 = coefficients_[3];
    const double k3 = coefficients_.size() >= 5 ? coefficients_[4] : 0; // K3 is allowed to be zero
    
    const double x = p.x();
    const double y = p.y();

    const double r2 = std::pow(x, 2) + std::pow(y, 2);
    const double r4 = std::pow(r2, 2);
    const double r6 = std::pow(r2, 3);

    // Tangential distortion
    const double tangential_x = (2 * p1 * x * y + p2 * (r2 + 2 * x*x));
    const double tangential_y = (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

    // Radial distortion
    double radial = (1 + k1*r2 + k2*r4 + k3*r6);

    // thin prisim distortion
    double prisim_x = 0;
    double prisim_y = 0;

    // If the calibration has k4, k5, k6
    if (coefficients_.size() >= 8)
    {
        const double& k4 = coefficients_[5];
        const double& k5 = coefficients_[6];
        const double& k6 = coefficients_[7];

        radial = radial / (1 + k4*r2 + k5*r4 + k6*r6);

        // If the calibration has s1, s2, s3, s4 (thin prisim distortion)
        if (coefficients_.size() >= 12)
        {
            const double& s1 = coefficients_[8];
            const double& s2 = coefficients_[9];
            const double& s3 = coefficients_[10];
            const double& s4 = coefficients_[11];

            prisim_x = s1*r2 + s2*r4;
            prisim_y = s3*r2 + s4*r4;

            if (coefficients_.size() == 14)
            {
                lvr2::panic_unimplemented("[PinholeModel] distortPoints - opencv model: Support for distortion coefficients 13 and 14 is not implemented!");
            }
        }
    }

    const double x_dist = x * radial + tangential_x + prisim_x;
    const double y_dist = y * radial + tangential_y + prisim_y;

    return lvr2::Vector2d(x_dist, y_dist);
}

std::string OpenCVDistortion::name() const
{
    return type;
}

const std::vector<double>& OpenCVDistortion::coefficients() const
{
    return coefficients_;
}

Vector2d UnknownDistortion::distortPoint(const Vector2d& p) const
{
    
    return p;
}

std::string UnknownDistortion::name() const
{
    return type;
}

const std::vector<double>& UnknownDistortion::coefficients() const
{
    return coefficients_;
}

} // namespace lvr2
