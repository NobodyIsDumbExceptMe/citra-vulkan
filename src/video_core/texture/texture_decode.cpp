// Copyright 2017 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <bit>
#include "common/assert.h"
#include "common/color.h"
#include "common/logging/log.h"
#include "common/swap.h"
#include "common/vector_math.h"
#include "video_core/regs_texturing.h"
#include "video_core/texture/etc1.h"
#include "video_core/texture/texture_decode.h"
#include "video_core/utils.h"

using TextureFormat = Pica::TexturingRegs::TextureFormat;

namespace Pica::Texture {

constexpr std::size_t TILE_SIZE = 8 * 8;
constexpr std::size_t ETC1_SUBTILES = 2 * 2;

size_t CalculateTileSize(TextureFormat format) {
    switch (format) {
    case TextureFormat::RGBA8:
        return 4 * TILE_SIZE;

    case TextureFormat::RGB8:
        return 3 * TILE_SIZE;

    case TextureFormat::RGB5A1:
    case TextureFormat::RGB565:
    case TextureFormat::RGBA4:
    case TextureFormat::IA8:
    case TextureFormat::RG8:
        return 2 * TILE_SIZE;

    case TextureFormat::I8:
    case TextureFormat::A8:
    case TextureFormat::IA4:
        return 1 * TILE_SIZE;

    case TextureFormat::I4:
    case TextureFormat::A4:
        return TILE_SIZE / 2;

    case TextureFormat::ETC1:
        return ETC1_SUBTILES * 8;

    case TextureFormat::ETC1A4:
        return ETC1_SUBTILES * 16;

    default: // placeholder for yet unknown formats
        UNIMPLEMENTED();
        return 0;
    }
}

Common::Vec4<u8> LookupTexture(const u8* source, unsigned int x, unsigned int y,
                               const TextureInfo& info, bool disable_alpha) {
    // Coordinate in tiles
    const unsigned int coarse_x = x / 8;
    const unsigned int coarse_y = y / 8;

    // Coordinate inside the tile
    const unsigned int fine_x = x % 8;
    const unsigned int fine_y = y % 8;

    const u8* line = source + coarse_y * info.stride;
    const u8* tile = line + coarse_x * CalculateTileSize(info.format);
    return LookupTexelInTile(tile, fine_x, fine_y, info, disable_alpha);
}

Common::Vec4<u8> LookupTexelInTile(const u8* source, unsigned int x, unsigned int y,
                                   const TextureInfo& info, bool disable_alpha) {
    DEBUG_ASSERT(x < 8);
    DEBUG_ASSERT(y < 8);

    using VideoCore::MortonInterleave;

    switch (info.format) {
    case TextureFormat::RGBA8: {
        auto res = Color::DecodeRGBA8(source + MortonInterleave(x, y) * 4);
        return {res.r(), res.g(), res.b(), static_cast<u8>(disable_alpha ? 255 : res.a())};
    }

    case TextureFormat::RGB8: {
        auto res = Color::DecodeRGB8(source + MortonInterleave(x, y) * 3);
        return {res.r(), res.g(), res.b(), 255};
    }

    case TextureFormat::RGB5A1: {
        auto res = Color::DecodeRGB5A1(source + MortonInterleave(x, y) * 2);
        return {res.r(), res.g(), res.b(), static_cast<u8>(disable_alpha ? 255 : res.a())};
    }

    case TextureFormat::RGB565: {
        auto res = Color::DecodeRGB565(source + MortonInterleave(x, y) * 2);
        return {res.r(), res.g(), res.b(), 255};
    }

    case TextureFormat::RGBA4: {
        auto res = Color::DecodeRGBA4(source + MortonInterleave(x, y) * 2);
        return {res.r(), res.g(), res.b(), static_cast<u8>(disable_alpha ? 255 : res.a())};
    }

    case TextureFormat::IA8: {
        const u8* source_ptr = source + MortonInterleave(x, y) * 2;

        if (disable_alpha) {
            // Show intensity as red, alpha as green
            return {source_ptr[1], source_ptr[0], 0, 255};
        } else {
            return {source_ptr[1], source_ptr[1], source_ptr[1], source_ptr[0]};
        }
    }

    case TextureFormat::RG8: {
        auto res = Color::DecodeRG8(source + MortonInterleave(x, y) * 2);
        return {res.r(), res.g(), 0, 255};
    }

    case TextureFormat::I8: {
        const u8* source_ptr = source + MortonInterleave(x, y);
        return {*source_ptr, *source_ptr, *source_ptr, 255};
    }

    case TextureFormat::A8: {
        const u8* source_ptr = source + MortonInterleave(x, y);

        if (disable_alpha) {
            return {*source_ptr, *source_ptr, *source_ptr, 255};
        } else {
            return {0, 0, 0, *source_ptr};
        }
    }

    case TextureFormat::IA4: {
        const u8* source_ptr = source + MortonInterleave(x, y);

        u8 i = Color::Convert4To8(((*source_ptr) & 0xF0) >> 4);
        u8 a = Color::Convert4To8((*source_ptr) & 0xF);

        if (disable_alpha) {
            // Show intensity as red, alpha as green
            return {i, a, 0, 255};
        } else {
            return {i, i, i, a};
        }
    }

    case TextureFormat::I4: {
        u32 morton_offset = MortonInterleave(x, y);
        const u8* source_ptr = source + morton_offset / 2;

        u8 i = (morton_offset % 2) ? ((*source_ptr & 0xF0) >> 4) : (*source_ptr & 0xF);
        i = Color::Convert4To8(i);

        return {i, i, i, 255};
    }

    case TextureFormat::A4: {
        u32 morton_offset = MortonInterleave(x, y);
        const u8* source_ptr = source + morton_offset / 2;

        u8 a = (morton_offset % 2) ? ((*source_ptr & 0xF0) >> 4) : (*source_ptr & 0xF);
        a = Color::Convert4To8(a);

        if (disable_alpha) {
            return {a, a, a, 255};
        } else {
            return {0, 0, 0, a};
        }
    }

    case TextureFormat::ETC1:
    case TextureFormat::ETC1A4: {
        bool has_alpha = (info.format == TextureFormat::ETC1A4);
        std::size_t subtile_size = has_alpha ? 16 : 8;

        // ETC1 further subdivides each 8x8 tile into four 4x4 subtiles
        constexpr unsigned int subtile_width = 4;
        constexpr unsigned int subtile_height = 4;

        unsigned int subtile_index = (x / subtile_width) + 2 * (y / subtile_height);
        x %= subtile_width;
        y %= subtile_height;

        const u8* subtile_ptr = source + subtile_index * subtile_size;

        u8 alpha = 255;
        if (has_alpha) {
            u64_le packed_alpha;
            memcpy(&packed_alpha, subtile_ptr, sizeof(u64));
            subtile_ptr += sizeof(u64);

            alpha = Color::Convert4To8((packed_alpha >> (4 * (x * subtile_width + y))) & 0xF);
        }

        u64_le subtile_data;
        memcpy(&subtile_data, subtile_ptr, sizeof(u64));

        return Common::MakeVec(SampleETC1Subtile(subtile_data, x, y),
                               disable_alpha ? (u8)255 : alpha);
    }

    default:
        LOG_ERROR(HW_GPU, "Unknown texture format: {:x}", (u32)info.format);
        DEBUG_ASSERT(false);
        return {};
    }
}

TextureInfo TextureInfo::FromPicaRegister(const TexturingRegs::TextureConfig& config,
                                          const TexturingRegs::TextureFormat& format) {
    TextureInfo info;
    info.physical_address = config.GetPhysicalAddress();
    info.width = config.width;
    info.height = config.height;
    info.format = format;
    info.SetDefaultStride();
    return info;
}

void ConvertBGRToRGB(std::span<const std::byte> source, std::span<std::byte> dest) {
    for (std::size_t i = 0; i < source.size(); i += 3) {
        u32 bgr{};
        std::memcpy(&bgr, source.data() + i, 3);
        const u32 rgb = Common::swap32(bgr << 8);
        std::memcpy(dest.data(), &rgb, 3);
    }
}

void ConvertBGRToRGBA(std::span<const std::byte> source, std::span<std::byte> dest) {
    u32 j = 0;
    for (std::size_t i = 0; i < source.size(); i += 3) {
        dest[j] = source[i + 2];
        dest[j + 1] = source[i + 1];
        dest[j + 2] = source[i];
        dest[j + 3] = std::byte{0xFF};
        j += 4;
    }
}

void ConvertABGRToRGBA(std::span<const std::byte> source, std::span<std::byte> dest) {
    for (u32 i = 0; i < source.size(); i += 4) {
        const u32 abgr = *reinterpret_cast<const u32*>(source.data() + i);
        const u32 rgba = Common::swap32(abgr);
        std::memcpy(dest.data() + i, &rgba, 4);
    }
}

} // namespace Pica::Texture
