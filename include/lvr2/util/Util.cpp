#include <algorithm>

#include <lvr2/util/Util.hpp>

namespace lvr2
{

int Util::getSpectralChannel(int wavelength, PointBuffer2Ptr pcloud, int fallback)
{
    FloatChannelOptional spectral_channels = pcloud->getFloatChannel("spectral_channels");
    if (!spectral_channels)
    {
        return fallback;
    }

    int minWavelength = *pcloud->getIntAttribute("spectral_wavelength_min");

    int channel = (wavelength - minWavelength) / wavelengthPerChannel(pcloud);

    if (channel < 0 || channel >= spectral_channels->width())
    {
        return fallback;
    }

    return channel;
}

int Util::getSpectralWavelength(int channel, PointBuffer2Ptr pcloud, int fallback)
{
    FloatChannelOptional spectral_channels = pcloud->getFloatChannel("spectral_channels");
    if (!spectral_channels)
    {
        return fallback;
    }
    
    int minWavelength = *pcloud->getIntAttribute("spectral_wavelength_min");

    if (channel < 0 || channel >= spectral_channels->width())
    {
        return fallback;
    }

    return channel * wavelengthPerChannel(pcloud) + minWavelength;
}

float Util::wavelengthPerChannel(PointBuffer2Ptr pcloud)
{
    FloatChannelOptional spectral_channels = pcloud->getFloatChannel("spectral_channels");
    if (!spectral_channels)
    {
        return -1.0f;
    }

    int minWavelength = *pcloud->getIntAttribute("spectral_wavelength_min");
    int maxWavelength = *pcloud->getIntAttribute("spectral_wavelength_max");

    return (maxWavelength - minWavelength) / static_cast<float>(spectral_channels->width());
}

} // namespace lvr2
