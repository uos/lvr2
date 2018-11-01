#include <algorithm>

#include <lvr2/util/Util.hpp>

namespace lvr2
{

int Util::getSpectralChannel(int wavelength, PointBufferPtr p, int fallback)
{
    FloatChannelOptional spectral_channels = p->getFloatChannel("spectral_channels");
    if (!spectral_channels)
    {
        return fallback;
    }

    int minWavelength = *p->getIntAttribute("spectral_wavelength_min");

    int channel = (wavelength - minWavelength) / wavelengthPerChannel(p);

    if (channel < 0 || channel >= spectral_channels->width())
    {
        return fallback;
    }

    return channel;
}

int Util::getSpectralWavelength(int channel, PointBufferPtr p, int fallback)
{
    FloatChannelOptional spectral_channels = p->getFloatChannel("spectral_channels");
    if (!spectral_channels)
    {
        return fallback;
    }
    
    int minWavelength = *p->getIntAttribute("spectral_wavelength_min");

    if (channel < 0 || channel >= spectral_channels->width())
    {
        return fallback;
    }

    return channel * wavelengthPerChannel(p) + minWavelength;
}

float Util::wavelengthPerChannel(PointBufferPtr p)
{
    FloatChannelOptional spectral_channels = p->getFloatChannel("spectral_channels");
    if (!spectral_channels)
    {
        return -1.0f;
    }

    int minWavelength = *p->getIntAttribute("spectral_wavelength_min");
    int maxWavelength = *p->getIntAttribute("spectral_wavelength_max");

    return (maxWavelength - minWavelength) / static_cast<float>(spectral_channels->width());
}

} // namespace lvr2
