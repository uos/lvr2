#pragma once

/**
 * @file tsdf.h
 * @author Marcel Flottmann
 * @date 2021-01-13
 */

#include <cinttypes>

struct TSDFValueHW
{
    using ValueType = int16_t;
    using WeightType = int16_t;

    ValueType value;
    WeightType weight;
};

class TSDFValue
{
public:
    using RawType = uint32_t;
    using ValueType = TSDFValueHW::ValueType;
    using WeightType = TSDFValueHW::WeightType;

private:
    union
    {
        RawType raw;
        TSDFValueHW tsdf;
    } data;
    static_assert(sizeof(RawType) == sizeof(TSDFValueHW));            // raw and TSDF types must be of the same size

public:

    TSDFValue(ValueType value, WeightType weight)
    {
        data.tsdf.value = value;
        data.tsdf.weight = weight;
    }

    explicit TSDFValue(RawType raw)
    {
        data.raw = raw;
    }

    TSDFValue() = default;
    TSDFValue(const TSDFValue&) = default;
    ~TSDFValue() = default;
    TSDFValue& operator=(const TSDFValue&) = default;

    bool operator==(const TSDFValue& rhs) const
    {
        return raw() == rhs.raw();
    }

    RawType raw() const
    {
        return data.raw;
    }

    void raw(RawType val)
    {
        data.raw = val;
    }

    ValueType value() const
    {
        return data.tsdf.value;
    }

    void value(ValueType val)
    {
        data.tsdf.value = val;
    }

    WeightType weight() const
    {
        return data.tsdf.weight;
    }

    void weight(WeightType val)
    {
        data.tsdf.weight = val;
    }
};

static_assert(sizeof(TSDFValue) == sizeof(TSDFValueHW));          // HW and SW types must be of the same size

