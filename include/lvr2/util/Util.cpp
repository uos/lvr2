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
    int maxWavelength = *pcloud->getIntAttribute("spectral_wavelength_max");

    float wavelengthPerChannel = (maxWavelength - minWavelength) / static_cast<float>(spectral_channels->width());
    int channel = (wavelength - minWavelength) / wavelengthPerChannel;

    if (channel < 0 || channel >= spectral_channels->width())
    {
        return fallback;
    }

    return channel;
}

} // namespace lvr2
