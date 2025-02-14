// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once
#include <span>
#include <bit>
#include <algorithm>
#include "common/alignment.h"
#include "common/color.h"
#include "video_core/rasterizer_cache/pixel_format.h"
#include "video_core/texture/etc1.h"
#include "video_core/utils.h"

namespace VideoCore {

template <typename T>
inline T MakeInt(const std::byte* bytes) {
    T integer{};
    std::memcpy(&integer, bytes, sizeof(T));

    return integer;
}

template <PixelFormat format>
inline void DecodePixel(const std::byte* source, std::byte* dest) {
    constexpr u32 bytes_per_pixel = GetFormatBpp(format) / 8;

    if constexpr (format == PixelFormat::D24S8) {
        const u32 d24s8 = std::rotl(MakeInt<u32>(source), 8);
        std::memcpy(dest, &d24s8, sizeof(u32));
    } else if constexpr (format == PixelFormat::IA8) {
        std::memset(dest, static_cast<int>(source[1]), 3);
        dest[3] = source[0];
    } else if constexpr (format == PixelFormat::RG8) {
        const auto rgba = Color::DecodeRG8(reinterpret_cast<const u8*>(source));
        std::memcpy(dest, rgba.AsArray(), 4);
    } else if constexpr (format == PixelFormat::I8) {
        std::memset(dest, static_cast<int>(source[0]), 3);
        dest[3] = std::byte{255};
    } else if constexpr (format == PixelFormat::A8) {
        std::memset(dest, 0, 3);
        dest[3] = source[0];
    } else if constexpr (format == PixelFormat::IA4) {
        const u8 ia4 = static_cast<const u8>(source[0]);
        std::memset(dest, Color::Convert4To8(ia4 >> 4), 3);
        dest[3] = std::byte{Color::Convert4To8(ia4 & 0xF)};
    } else {
        std::memcpy(dest, source, bytes_per_pixel);
    }
}

template <PixelFormat format>
inline void DecodePixel4(u32 x, u32 y, const std::byte* source_tile, std::byte* dest_pixel) {
    const u32 morton_offset = VideoCore::MortonInterleave(x, y);
    const u8 value = static_cast<const u8>(source_tile[morton_offset >> 1]);
    const u8 pixel = Color::Convert4To8((morton_offset % 2) ? (value >> 4) : (value & 0xF));

    if constexpr (format == PixelFormat::I4) {
        std::memset(dest_pixel, static_cast<int>(pixel), 3);
        dest_pixel[3] = std::byte{255};
    } else {
        std::memset(dest_pixel, 0, 3);
        dest_pixel[3] = std::byte{pixel};
    }
}

template <PixelFormat format>
inline void DecodePixelETC1(u32 x, u32 y, const std::byte* source_tile, std::byte* dest_pixel) {
    constexpr u32 subtile_width = 4;
    constexpr u32 subtile_height = 4;
    constexpr bool has_alpha = format == PixelFormat::ETC1A4;
    constexpr std::size_t subtile_size = has_alpha ? 16 : 8;

    const u32 subtile_index = (x / subtile_width) + 2 * (y / subtile_height);
    x %= subtile_width;
    y %= subtile_height;

    const std::byte* subtile_ptr = source_tile + subtile_index * subtile_size;

    u8 alpha = 255;
    if constexpr (has_alpha) {
        u64_le packed_alpha;
        std::memcpy(&packed_alpha, subtile_ptr, sizeof(u64));
        subtile_ptr += sizeof(u64);

        alpha = Color::Convert4To8((packed_alpha >> (4 * (x * subtile_width + y))) & 0xF);
    }

    const u64_le subtile_data = MakeInt<u64_le>(subtile_ptr);
    const auto rgb = Pica::Texture::SampleETC1Subtile(subtile_data, x, y);

    // Copy the uncompressed pixel to the destination
    std::memcpy(dest_pixel, rgb.AsArray(), 3);
    dest_pixel[3] = std::byte{alpha};
}

template <PixelFormat format>
inline void EncodePixel(const std::byte* source, std::byte* dest) {
    constexpr u32 bytes_per_pixel = GetFormatBpp(format) / 8;

    if constexpr (format == PixelFormat::D24S8) {
        const u32 s8d24 = std::rotr(MakeInt<u32>(source), 8);
        std::memcpy(dest, &s8d24, sizeof(u32));
    } else {
        std::memcpy(dest, source, bytes_per_pixel);
    }
}

template <bool morton_to_linear, PixelFormat format>
inline void MortonCopyTile(u32 stride, std::span<std::byte> tile_buffer, std::span<std::byte> linear_buffer) {
    constexpr u32 bytes_per_pixel = GetFormatBpp(format) / 8;
    constexpr u32 linear_bytes_per_pixel = GetBytesPerPixel(format);
    constexpr bool is_compressed = format == PixelFormat::ETC1 || format == PixelFormat::ETC1A4;
    constexpr bool is_4bit = format == PixelFormat::I4 || format == PixelFormat::A4;

    for (u32 y = 0; y < 8; y++) {
        for (u32 x = 0; x < 8; x++) {
            const auto tiled_pixel = tile_buffer.subspan(VideoCore::MortonInterleave(x, y) * bytes_per_pixel,
                                                         bytes_per_pixel);
            const auto linear_pixel = linear_buffer.subspan(((7 - y) * stride + x) * linear_bytes_per_pixel,
                                                            linear_bytes_per_pixel);
            if constexpr (morton_to_linear) {
                if constexpr (is_compressed) {
                    DecodePixelETC1<format>(x, y, tile_buffer.data(), linear_pixel.data());
                } else if constexpr (is_4bit) {
                    DecodePixel4<format>(x, y, tile_buffer.data(), linear_pixel.data());
                } else {
                    DecodePixel<format>(tiled_pixel.data(), linear_pixel.data());
                }
            } else {
                EncodePixel<format>(linear_pixel.data(), tiled_pixel.data());
            }
        }
    }
}

template <bool morton_to_linear, PixelFormat format>
static void MortonCopy(u32 stride, u32 height, u32 start_offset, u32 end_offset,
                       std::span<std::byte> linear_buffer,
                       std::span<std::byte> tiled_buffer) {

    constexpr u32 bytes_per_pixel = GetFormatBpp(format) / 8;
    constexpr u32 aligned_bytes_per_pixel = GetBytesPerPixel(format);
    static_assert(aligned_bytes_per_pixel >= bytes_per_pixel, "");

    // We could use bytes_per_pixel here but it should be avoided because it
    // becomes zero for 4-bit textures!
    constexpr u32 tile_size = GetFormatBpp(format) * 64 / 8;
    const u32 linear_tile_size = (7 * stride + 8) * aligned_bytes_per_pixel;

    // Does this line have any significance?
    //u32 linear_offset = aligned_bytes_per_pixel - bytes_per_pixel;
    u32 linear_offset = 0;
    u32 tiled_offset = 0;

    const PAddr aligned_down_start_offset = Common::AlignDown(start_offset, tile_size);
    const PAddr aligned_start_offset = Common::AlignUp(start_offset, tile_size);
    PAddr aligned_end_offset = Common::AlignDown(end_offset, tile_size);

    ASSERT(!morton_to_linear || (aligned_start_offset == start_offset && aligned_end_offset == end_offset));

    const u32 begin_pixel_index = aligned_down_start_offset * 8 / GetFormatBpp(format);
    u32 x = (begin_pixel_index % (stride * 8)) / 8;
    u32 y = (begin_pixel_index / (stride * 8)) * 8;

    // In OpenGL the texture origin is in the bottom left corner as opposed to other
    // APIs that have it at the top left. To avoid flipping texture coordinates in
    // the shader we read/write the linear buffer backwards
    linear_offset += ((height - 8 - y) * stride + x) * aligned_bytes_per_pixel;

    auto linear_next_tile = [&] {
        x = (x + 8) % stride;
        linear_offset += 8 * aligned_bytes_per_pixel;
        if (!x) {
            y  = (y + 8) % height;
            if (!y) {
                return;
            }

            linear_offset -= stride * 9 * aligned_bytes_per_pixel;
        }
    };

    // If during a texture download the start coordinate is not tile aligned, swizzle
    // the tile affected to a temporary buffer and copy the part we are interested in
    if (start_offset < aligned_start_offset && !morton_to_linear) {
        std::array<std::byte, tile_size> tmp_buf;
        auto linear_data = linear_buffer.last(linear_buffer.size_bytes() - linear_offset);
        MortonCopyTile<morton_to_linear, format>(stride, tmp_buf, linear_data);

        std::memcpy(tiled_buffer.data(), tmp_buf.data() + start_offset - aligned_down_start_offset,
                    std::min(aligned_start_offset, end_offset) - start_offset);

        tiled_offset += aligned_start_offset - start_offset;
        linear_next_tile();
    }

    const u32 buffer_end = tiled_offset + aligned_end_offset - aligned_start_offset;
    while (tiled_offset < buffer_end) {
        auto linear_data = linear_buffer.last(linear_buffer.size_bytes() - linear_offset);
        auto tiled_data = tiled_buffer.subspan(tiled_offset, tile_size);
        MortonCopyTile<morton_to_linear, format>(stride, tiled_data, linear_data);
        tiled_offset += tile_size;
        linear_next_tile();
    }

    // If during a texture download the end coordinate is not tile aligned, swizzle
    // the tile affected to a temporary buffer and copy the part we are interested in
    if (end_offset > std::max(aligned_start_offset, aligned_end_offset) && !morton_to_linear) {
        std::array<std::byte, tile_size> tmp_buf;
        auto linear_data = linear_buffer.subspan(linear_offset, linear_tile_size);
        MortonCopyTile<morton_to_linear, format>(stride, tmp_buf, linear_data);
        std::memcpy(tiled_buffer.data() + tiled_offset, tmp_buf.data(), end_offset - aligned_end_offset);
    }
}

using MortonFunc = void (*)(u32, u32, u32, u32, std::span<std::byte>, std::span<std::byte>);

static constexpr std::array<MortonFunc, 18> UNSWIZZLE_TABLE = {
    MortonCopy<true, PixelFormat::RGBA8>,  // 0
    MortonCopy<true, PixelFormat::RGB8>,   // 1
    MortonCopy<true, PixelFormat::RGB5A1>, // 2
    MortonCopy<true, PixelFormat::RGB565>, // 3
    MortonCopy<true, PixelFormat::RGBA4>,  // 4
    MortonCopy<true, PixelFormat::IA8>,  // 5
    MortonCopy<true, PixelFormat::RG8>,  // 6
    MortonCopy<true, PixelFormat::I8>,  // 7
    MortonCopy<true, PixelFormat::A8>,  // 8
    MortonCopy<true, PixelFormat::IA4>,  // 9
    MortonCopy<true, PixelFormat::I4>,  // 10
    MortonCopy<true, PixelFormat::A4>,  // 11
    MortonCopy<true, PixelFormat::ETC1>,  // 12
    MortonCopy<true, PixelFormat::ETC1A4>,  // 13
    MortonCopy<true, PixelFormat::D16>,  // 14
    nullptr,                             // 15
    MortonCopy<true, PixelFormat::D24>,  // 16
    MortonCopy<true, PixelFormat::D24S8> // 17
};

static constexpr std::array<MortonFunc, 18> SWIZZLE_TABLE = {
    MortonCopy<false, PixelFormat::RGBA8>,  // 0
    MortonCopy<false, PixelFormat::RGB8>,   // 1
    MortonCopy<false, PixelFormat::RGB5A1>, // 2
    MortonCopy<false, PixelFormat::RGB565>, // 3
    MortonCopy<false, PixelFormat::RGBA4>,  // 4
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,                              // 5 - 13
    MortonCopy<false, PixelFormat::D16>,  // 14
    nullptr,                              // 15
    MortonCopy<false, PixelFormat::D24>,  // 16
    MortonCopy<false, PixelFormat::D24S8> // 17
};

} // namespace OpenGL
