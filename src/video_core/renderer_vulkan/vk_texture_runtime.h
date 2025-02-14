// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once
#include <span>
#include <set>
#include <vulkan/vulkan_hash.hpp>
#include "video_core/rasterizer_cache/rasterizer_cache.h"
#include "video_core/rasterizer_cache/surface_base.h"
#include "video_core/rasterizer_cache/types.h"
#include "video_core/renderer_vulkan/vk_stream_buffer.h"
#include "video_core/renderer_vulkan/vk_task_scheduler.h"

namespace Vulkan {

struct StagingData {
    vk::Buffer buffer;
    u32 size = 0;
    std::span<std::byte> mapped{};
    u32 buffer_offset = 0;
};

struct ImageAlloc {
    vk::Image image;
    vk::ImageView image_view;
    VmaAllocation allocation;
    vk::ImageLayout layout = vk::ImageLayout::eUndefined;
    vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eNone;
    u32 levels = 1;
};

class Instance;
class RenderpassCache;
class Surface;

/**
 * Provides texture manipulation functions to the rasterizer cache
 * Separating this into a class makes it easier to abstract graphics API code
 */
class TextureRuntime {
    friend class Surface;
public:
    TextureRuntime(const Instance& instance, TaskScheduler& scheduler,
                   RenderpassCache& renderpass_cache);
    ~TextureRuntime();

    /// Maps an internal staging buffer of the provided size of pixel uploads/downloads
    [[nodiscard]] StagingData FindStaging(u32 size, bool upload);

    /// Allocates a vulkan image possibly resusing an existing one
    [[nodiscard]] ImageAlloc Allocate(u32 width, u32 height, VideoCore::PixelFormat format,
                                      VideoCore::TextureType type);

    /// Takes back ownership of the allocation for recycling
    void Recycle(const VideoCore::HostTextureTag tag, ImageAlloc&& alloc);

    /// Performs required format convertions on the staging data
    void FormatConvert(VideoCore::PixelFormat format,  bool upload,
                       std::span<std::byte> source, std::span<std::byte> dest);

    /// Transitions the mip level range of the surface to new_layout
    void Transition(vk::CommandBuffer command_buffer, ImageAlloc& alloc,
                    vk::ImageLayout new_layout, u32 level, u32 level_count,
                    u32 layer = 0, u32 layer_count = 1);

    /// Fills the rectangle of the texture with the clear value provided
    bool ClearTexture(Surface& surface, const VideoCore::TextureClear& clear,
                      VideoCore::ClearValue value);

    /// Copies a rectangle of src_tex to another rectange of dst_rect
    bool CopyTextures(Surface& source, Surface& dest, const VideoCore::TextureCopy& copy);

    /// Blits a rectangle of src_tex to another rectange of dst_rect
    bool BlitTextures(Surface& surface, Surface& dest, const VideoCore::TextureBlit& blit);

    /// Generates mipmaps for all the available levels of the texture
    void GenerateMipmaps(Surface& surface, u32 max_level);

    /// Performs operations that need to be done on every scheduler slot switch
    void OnSlotSwitch(u32 new_slot);

private:
    /// Returns the current Vulkan instance
    const Instance& GetInstance() const {
        return instance;
    }

    /// Returns the current Vulkan scheduler
    TaskScheduler& GetScheduler() const {
        return scheduler;
    }

private:
    const Instance& instance;
    TaskScheduler& scheduler;
    RenderpassCache& renderpass_cache;
    std::array<std::unique_ptr<StagingBuffer>, SCHEDULER_COMMAND_COUNT> staging_buffers;
    std::array<u32, SCHEDULER_COMMAND_COUNT> staging_offsets{};
    std::unordered_multimap<VideoCore::HostTextureTag, ImageAlloc> texture_recycler;
    std::unordered_map<vk::ImageView, vk::Framebuffer> clear_framebuffers;
};

class Surface : public VideoCore::SurfaceBase<Surface> {
    friend class TextureRuntime;
    friend class RasterizerVulkan;
public:
    Surface(VideoCore::SurfaceParams& params, TextureRuntime& runtime);
    ~Surface() override;

    /// Uploads pixel data in staging to a rectangle region of the surface texture
    void Upload(const VideoCore::BufferTextureCopy& upload, const StagingData& staging);

    /// Downloads pixel data to staging from a rectangle region of the surface texture
    void Download(const VideoCore::BufferTextureCopy& download, const StagingData& staging);

private:
    /// Downloads scaled image by downscaling the requested rectangle
    void ScaledDownload(const VideoCore::BufferTextureCopy& download);

    /// Uploads pixel data to scaled texture
    void ScaledUpload(const VideoCore::BufferTextureCopy& upload);

    /// Overrides the image layout of the mip level range
    void SetLayout(vk::ImageLayout new_layout, u32 level = 0, u32 level_count = 1);

private:
    TextureRuntime& runtime;
    const Instance& instance;
    TaskScheduler& scheduler;

    ImageAlloc alloc{};
    vk::Format internal_format = vk::Format::eUndefined;
};

struct Traits {
    using RuntimeType = TextureRuntime;
    using SurfaceType = Surface;
};

using RasterizerCache = VideoCore::RasterizerCache<Traits>;

} // namespace Vulkan
