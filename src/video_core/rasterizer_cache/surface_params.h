// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <climits>
#include <boost/icl/interval_map.hpp>
#include <boost/icl/interval_set.hpp>
#include "common/math_util.h"
#include "video_core/rasterizer_cache/pixel_format.h"

namespace VideoCore {

using SurfaceInterval = boost::icl::right_open_interval<PAddr>;

enum class TextureType {
    Texture2D = 0,
    CubeMap = 1
};

class SurfaceParams {
public:
    /// Surface match traits
    bool ExactMatch(const SurfaceParams& other_surface) const;
    bool CanSubRect(const SurfaceParams& sub_surface) const;
    bool CanExpand(const SurfaceParams& expanded_surface) const;
    bool CanTexCopy(const SurfaceParams& texcopy_params) const;

    Common::Rectangle<u32> GetSubRect(const SurfaceParams& sub_surface) const;
    Common::Rectangle<u32> GetScaledSubRect(const SurfaceParams& sub_surface) const;

    /// Returns the outer rectangle containing "interval"
    SurfaceParams FromInterval(SurfaceInterval interval) const;
    SurfaceInterval GetSubRectInterval(Common::Rectangle<u32> unscaled_rect) const;

    /// Updates remaining members from the already set addr, width, height and pixel_format
    void UpdateParams() {
        if (stride == 0) {
            stride = width;
        }

        type = GetFormatType(pixel_format);
        size = !is_tiled ? BytesInPixels(stride * (height - 1) + width)
                         : BytesInPixels(stride * 8 * (height / 8 - 1) + width * 8);
        end = addr + size;
    }

    bool IsScaled() const {
        return res_scale > 1;
    }

    SurfaceInterval GetInterval() const {
        return SurfaceInterval(addr, end);
    }

    u32 GetFormatBpp() const {
        return VideoCore::GetFormatBpp(pixel_format);
    }

    u32 GetScaledWidth() const {
        return width * res_scale;
    }

    u32 GetScaledHeight() const {
        return height * res_scale;
    }

    Common::Rectangle<u32> GetRect() const {
        return {0, height, width, 0};
    }

    Common::Rectangle<u32> GetScaledRect() const {
        return {0, GetScaledHeight(), GetScaledWidth(), 0};
    }

    u32 PixelsInBytes(u32 size) const {
        return size * 8 / GetFormatBpp();
    }

    u32 BytesInPixels(u32 pixels) const {
        return pixels * GetFormatBpp() / 8;
    }

public:
    PAddr addr = 0;
    PAddr end = 0;
    u32 size = 0;

    u32 width = 0;
    u32 height = 0;
    u32 stride = 0;
    u16 res_scale = 1;

    bool is_tiled = false;
    TextureType texture_type = TextureType::Texture2D;
    PixelFormat pixel_format = PixelFormat::Invalid;
    SurfaceType type = SurfaceType::Invalid;
};

} // namespace OpenGL
