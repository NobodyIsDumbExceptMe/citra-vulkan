// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/scope_exit.h"
#include "video_core/rasterizer_cache/utils.h"
#include "video_core/renderer_opengl/gl_texture_runtime.h"
#include "video_core/renderer_opengl/gl_driver.h"
#include "video_core/renderer_opengl/gl_state.h"

namespace OpenGL {

constexpr FormatTuple DEFAULT_TUPLE = {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE};

static constexpr std::array DEPTH_TUPLES = {
    FormatTuple{GL_DEPTH_COMPONENT16, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT}, // D16
    FormatTuple{},
    FormatTuple{GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT},   // D24
    FormatTuple{GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8}, // D24S8
};

static constexpr std::array COLOR_TUPLES = {
    FormatTuple{GL_RGBA8, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8},     // RGBA8
    FormatTuple{GL_RGB8, GL_BGR, GL_UNSIGNED_BYTE},              // RGB8
    FormatTuple{GL_RGB5_A1, GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1}, // RGB5A1
    FormatTuple{GL_RGB565, GL_RGB, GL_UNSIGNED_SHORT_5_6_5},     // RGB565
    FormatTuple{GL_RGBA4, GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4},   // RGBA4
};

static constexpr std::array COLOR_TUPLES_OES = {
    FormatTuple{GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE},            // RGBA8
    FormatTuple{GL_RGB8, GL_RGB, GL_UNSIGNED_BYTE},              // RGB8
    FormatTuple{GL_RGB5_A1, GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1}, // RGB5A1
    FormatTuple{GL_RGB565, GL_RGB, GL_UNSIGNED_SHORT_5_6_5},     // RGB565
    FormatTuple{GL_RGBA4, GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4},   // RGBA4
};

GLbitfield MakeBufferMask(VideoCore::SurfaceType type) {
    switch (type) {
    case VideoCore::SurfaceType::Color:
    case VideoCore::SurfaceType::Texture:
    case VideoCore::SurfaceType::Fill:
        return GL_COLOR_BUFFER_BIT;
    case VideoCore::SurfaceType::Depth:
        return GL_DEPTH_BUFFER_BIT;
    case VideoCore::SurfaceType::DepthStencil:
        return GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT;
    default:
        UNREACHABLE_MSG("Invalid surface type!");
    }

    return GL_COLOR_BUFFER_BIT;
}

TextureRuntime::TextureRuntime(Driver& driver)
    : driver{driver}, downloader_es{false},
      filterer{Settings::values.texture_filter_name, VideoCore::GetResolutionScaleFactor()} {

    read_fbo.Create();
    draw_fbo.Create();
}

const StagingBuffer& TextureRuntime::FindStaging(u32 size, bool upload) {
    const GLenum target = upload ? GL_PIXEL_UNPACK_BUFFER : GL_PIXEL_PACK_BUFFER;
    const GLbitfield access = upload ? GL_MAP_WRITE_BIT : GL_MAP_READ_BIT;
    auto& search = upload ? upload_buffers : download_buffers;

    // Attempt to find a free buffer that fits the requested data
    for (auto it = search.lower_bound({.size = size}); it != search.end(); it++) {
        if (!upload || it->IsFree()) {
            return *it;
        }
    }

    OGLBuffer buffer{};
    buffer.Create();

    glBindBuffer(target, buffer.handle);

    // Allocate a new buffer and map the data to the host
    std::byte* data = nullptr;
    if (driver.IsOpenGLES() && driver.HasExtBufferStorage()) {
        const GLbitfield storage = upload ? GL_MAP_WRITE_BIT : GL_MAP_READ_BIT | GL_CLIENT_STORAGE_BIT_EXT;
        glBufferStorageEXT(target, size, nullptr, storage | GL_MAP_PERSISTENT_BIT_EXT |
                                                            GL_MAP_COHERENT_BIT_EXT);
        data = reinterpret_cast<std::byte*>(glMapBufferRange(target, 0, size, access | GL_MAP_PERSISTENT_BIT_EXT |
                                                          GL_MAP_COHERENT_BIT_EXT));
    } else if (driver.HasArbBufferStorage()) {
        const GLbitfield storage = upload ? GL_MAP_WRITE_BIT : GL_MAP_READ_BIT | GL_CLIENT_STORAGE_BIT;
        glBufferStorage(target, size, nullptr, storage | GL_MAP_PERSISTENT_BIT |
                                                         GL_MAP_COHERENT_BIT);
        data = reinterpret_cast<std::byte*>(glMapBufferRange(target, 0, size, access | GL_MAP_PERSISTENT_BIT |
                                                          GL_MAP_COHERENT_BIT));
    } else {
        UNIMPLEMENTED();
    }

    glBindBuffer(target, 0);

    StagingBuffer staging = {
        .buffer = std::move(buffer),
        .mapped = std::span{data, size},
        .size = size
    };

    const auto& it = search.emplace(std::move(staging));
    return *it;
}

const FormatTuple& TextureRuntime::GetFormatTuple(VideoCore::PixelFormat pixel_format) {
    const auto type = GetFormatType(pixel_format);
    const std::size_t format_index = static_cast<std::size_t>(pixel_format);

    if (type == VideoCore::SurfaceType::Color) {
        ASSERT(format_index < COLOR_TUPLES.size());
        return (driver.IsOpenGLES() ? COLOR_TUPLES_OES : COLOR_TUPLES)[format_index];
    } else if (type == VideoCore::SurfaceType::Depth ||
               type == VideoCore::SurfaceType::DepthStencil) {
        const std::size_t tuple_idx = format_index - 14;
        ASSERT(tuple_idx < DEPTH_TUPLES.size());
        return DEPTH_TUPLES[tuple_idx];
    }

    return DEFAULT_TUPLE;
}

void TextureRuntime::FormatConvert(VideoCore::PixelFormat format,  bool upload,
                                   std::span<std::byte> source, std::span<std::byte> dest) {
    if (format == VideoCore::PixelFormat::RGBA8 && driver.IsOpenGLES()) {
        Pica::Texture::ConvertABGRToRGBA(source, dest);
    } else if (format == VideoCore::PixelFormat::RGB8 && driver.IsOpenGLES()) {
        Pica::Texture::ConvertBGRToRGB(source, dest);
    } else {
        std::memcpy(dest.data(), source.data(), source.size());
    }
}

OGLTexture TextureRuntime::Allocate(u32 width, u32 height, VideoCore::PixelFormat format,
                                    VideoCore::TextureType type) {

    const u32 layers = type == VideoCore::TextureType::CubeMap ? 6 : 1;
    const GLenum target =
            type == VideoCore::TextureType::CubeMap ? GL_TEXTURE_CUBE_MAP : GL_TEXTURE_2D;
    const VideoCore::HostTextureTag key = {
        .format = format,
        .width = width,
        .height = height,
        .layers = layers
    };

    // Attempt to recycle an unused texture
    if (auto it = texture_recycler.find(key); it != texture_recycler.end()) {
        OGLTexture texture = std::move(it->second);
        texture_recycler.erase(it);
        return texture;
    }

    const auto& tuple = GetFormatTuple(format);
    const OpenGLState& state = OpenGLState::GetCurState();
    GLuint old_tex = state.texture_units[0].texture_2d;

    // Allocate new texture
    OGLTexture texture{};
    texture.Create();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(target, texture.handle);

    glTexStorage2D(target, std::bit_width(std::max(width, height)),
                   tuple.internal_format, width, height);

    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glBindTexture(target, old_tex);
    return texture;
}

bool TextureRuntime::ClearTexture(Surface& surface, const VideoCore::TextureClear& clear,
                                  VideoCore::ClearValue value) {
    OpenGLState prev_state = OpenGLState::GetCurState();
    SCOPE_EXIT({ prev_state.Apply(); });

    // Setup scissor rectangle according to the clear rectangle
    OpenGLState state{};
    state.scissor.enabled = true;
    state.scissor.x = clear.texture_rect.left;
    state.scissor.y = clear.texture_rect.bottom;
    state.scissor.width = clear.texture_rect.GetWidth();
    state.scissor.height = clear.texture_rect.GetHeight();
    state.draw.draw_framebuffer = draw_fbo.handle;
    state.Apply();

    GLint handle = surface.texture.handle;
    switch (surface.type) {
    case VideoCore::SurfaceType::Color:
    case VideoCore::SurfaceType::Texture:
    case VideoCore::SurfaceType::Fill:
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, handle,
                               clear.texture_level);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, 0,
                               0);

        state.color_mask.red_enabled = true;
        state.color_mask.green_enabled = true;
        state.color_mask.blue_enabled = true;
        state.color_mask.alpha_enabled = true;
        state.Apply();

        glClearBufferfv(GL_COLOR, 0, value.color.AsArray());
        break;
    case VideoCore::SurfaceType::Depth:
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, handle,
                               clear.texture_level);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_TEXTURE_2D, 0, 0);

        state.depth.write_mask = GL_TRUE;
        state.Apply();

        glClearBufferfv(GL_DEPTH, 0, &value.depth);
        break;
    case VideoCore::SurfaceType::DepthStencil:
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D,
                               handle, clear.texture_level);

        state.depth.write_mask = GL_TRUE;
        state.stencil.write_mask = -1;
        state.Apply();

        glClearBufferfi(GL_DEPTH_STENCIL, 0, value.depth, value.stencil);
        break;
    default:
        UNREACHABLE_MSG("Invalid surface type!");
    }

    return true;
}

bool TextureRuntime::CopyTextures(Surface& source, Surface& dest, const VideoCore::TextureCopy& copy) {
    return true;
}

bool TextureRuntime::BlitTextures(Surface& source, Surface& dest, const VideoCore::TextureBlit& blit) {
    OpenGLState prev_state = OpenGLState::GetCurState();
    SCOPE_EXIT({ prev_state.Apply(); });

    OpenGLState state{};
    state.draw.read_framebuffer = read_fbo.handle;
    state.draw.draw_framebuffer = draw_fbo.handle;
    state.Apply();

    const GLenum src_textarget = source.texture_type == VideoCore::TextureType::CubeMap ?
                GL_TEXTURE_CUBE_MAP_POSITIVE_X + blit.src_layer : GL_TEXTURE_2D;
    BindFramebuffer(GL_READ_FRAMEBUFFER, blit.src_level, src_textarget, source.type, source.texture);

    const GLenum dst_textarget = dest.texture_type == VideoCore::TextureType::CubeMap ?
                GL_TEXTURE_CUBE_MAP_POSITIVE_X + blit.dst_layer : GL_TEXTURE_2D;
    BindFramebuffer(GL_DRAW_FRAMEBUFFER, blit.dst_level, dst_textarget, dest.type, dest.texture);

    // TODO (wwylele): use GL_NEAREST for shadow map texture
    // Note: shadow map is treated as RGBA8 format in PICA, as well as in the rasterizer cache, but
    // doing linear intepolation componentwise would cause incorrect value. However, for a
    // well-programmed game this code path should be rarely executed for shadow map with
    // inconsistent scale.
    const GLbitfield buffer_mask = MakeBufferMask(source.type);
    const GLenum filter = buffer_mask == GL_COLOR_BUFFER_BIT ? GL_LINEAR : GL_NEAREST;
    glBlitFramebuffer(blit.src_rect.left, blit.src_rect.bottom, blit.src_rect.right, blit.src_rect.top,
                      blit.dst_rect.left, blit.dst_rect.bottom, blit.dst_rect.right, blit.dst_rect.top,
                      buffer_mask, filter);

    return true;
}

void TextureRuntime::GenerateMipmaps(Surface& surface, u32 max_level) {
    OpenGLState prev_state = OpenGLState::GetCurState();
    SCOPE_EXIT({ prev_state.Apply(); });

    OpenGLState state{};
    state.texture_units[0].texture_2d = surface.texture.handle;
    state.Apply();

    glActiveTexture(GL_TEXTURE0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, max_level);
    glGenerateMipmap(GL_TEXTURE_2D);
}

void TextureRuntime::BindFramebuffer(GLenum target, GLint level, GLenum textarget,
                                     VideoCore::SurfaceType type, OGLTexture& texture) const {
    const GLint framebuffer = target == GL_DRAW_FRAMEBUFFER ? draw_fbo.handle : read_fbo.handle;
    glBindFramebuffer(target, framebuffer);

    switch (type) {
    case VideoCore::SurfaceType::Color:
    case VideoCore::SurfaceType::Texture:
        glFramebufferTexture2D(target, GL_COLOR_ATTACHMENT0, textarget, texture.handle, level);
        break;
    case VideoCore::SurfaceType::Depth:
        glFramebufferTexture2D(target, GL_DEPTH_ATTACHMENT, textarget, texture.handle, level);
        break;
    case VideoCore::SurfaceType::DepthStencil:
        glFramebufferTexture2D(target, GL_DEPTH_STENCIL_ATTACHMENT, textarget, texture.handle, level);
        break;
    default:
        UNREACHABLE_MSG("Invalid surface type!");
    }
}

Surface::Surface(VideoCore::SurfaceParams& params, TextureRuntime& runtime)
    : VideoCore::SurfaceBase<Surface>{params}, runtime{runtime}, driver{runtime.GetDriver()} {
    texture = runtime.Allocate(GetScaledWidth(), GetScaledHeight(), params.pixel_format, texture_type);
}

Surface::~Surface() {
    const VideoCore::HostTextureTag tag = {
        .format = pixel_format,
        .width = GetScaledWidth(),
        .height = GetScaledHeight(),
        .layers = texture_type == VideoCore::TextureType::CubeMap ? 6u : 1u
    };

    runtime.texture_recycler.emplace(tag, std::move(texture));
}

MICROPROFILE_DEFINE(OpenGL_Upload, "OpenGLSurface", "Texture Upload", MP_RGB(128, 192, 64));
void Surface::Upload(const VideoCore::BufferTextureCopy& upload, const StagingBuffer& staging) {
    MICROPROFILE_SCOPE(OpenGL_Upload);

    // Ensure no bad interactions with GL_UNPACK_ALIGNMENT
    ASSERT(stride * GetBytesPerPixel(pixel_format) % 4 == 0);

    OpenGLState prev_state = OpenGLState::GetCurState();
    SCOPE_EXIT({ prev_state.Apply(); });

    glPixelStorei(GL_UNPACK_ROW_LENGTH, static_cast<GLint>(stride));
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, staging.buffer.handle);

    const bool is_scaled = res_scale != 1;
    if (is_scaled) {
        ScaledUpload(upload);
    } else {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture.handle);

        const auto& tuple = runtime.GetFormatTuple(pixel_format);
        glTexSubImage2D(GL_TEXTURE_2D, upload.texture_level,
                        upload.texture_rect.left, upload.texture_rect.bottom,
                        upload.texture_rect.GetWidth(),
                        upload.texture_rect.GetHeight(),
                        tuple.format, tuple.type, 0);
    }

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);

    // Lock the staging buffer until glTexSubImage completes
    staging.Lock();
    InvalidateAllWatcher();
}

MICROPROFILE_DEFINE(OpenGL_Download, "OpenGLSurface", "Texture Download", MP_RGB(128, 192, 64));
void Surface::Download(const VideoCore::BufferTextureCopy& download, const StagingBuffer& staging) {
    MICROPROFILE_SCOPE(OpenGL_Download);

    // Ensure no bad interactions with GL_PACK_ALIGNMENT
    ASSERT(stride * GetBytesPerPixel(pixel_format) % 4 == 0);

    OpenGLState prev_state = OpenGLState::GetCurState();
    SCOPE_EXIT({ prev_state.Apply(); });

    glPixelStorei(GL_PACK_ROW_LENGTH, static_cast<GLint>(stride));
    glBindBuffer(GL_PIXEL_PACK_BUFFER, staging.buffer.handle);

    const bool is_scaled = res_scale != 1;
    if (is_scaled) {
        ScaledDownload(download);
    } else {
        runtime.BindFramebuffer(GL_READ_FRAMEBUFFER, download.texture_level, GL_TEXTURE_2D, type, texture);

        const auto& tuple = runtime.GetFormatTuple(pixel_format);
        glReadPixels(download.texture_rect.left, download.texture_rect.bottom,
                     download.texture_rect.GetWidth(), download.texture_rect.GetHeight(),
                     tuple.format, tuple.type, 0);
    }

    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    glPixelStorei(GL_PACK_ROW_LENGTH, 0);
}

void Surface::ScaledDownload(const VideoCore::BufferTextureCopy& download) {
    const u32 rect_width = download.texture_rect.GetWidth();
    const u32 rect_height = download.texture_rect.GetHeight();

    // Allocate an unscaled texture that fits the download rectangle to use as a blit destination
    OGLTexture unscaled_tex = runtime.Allocate(rect_width, rect_height, pixel_format,
                                               VideoCore::TextureType::Texture2D);
    runtime.BindFramebuffer(GL_DRAW_FRAMEBUFFER, 0, GL_TEXTURE_2D, type, unscaled_tex);
    runtime.BindFramebuffer(GL_READ_FRAMEBUFFER, download.texture_level, GL_TEXTURE_2D, type, texture);

    // Blit the scaled rectangle to the unscaled texture
    const VideoCore::Rect2D scaled_rect = download.texture_rect * res_scale;
    glBlitFramebuffer(scaled_rect.left, scaled_rect.bottom, scaled_rect.right, scaled_rect.top,
                      0, 0, rect_width, rect_height, MakeBufferMask(type), GL_LINEAR);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, unscaled_tex.handle);

    const auto& tuple = runtime.GetFormatTuple(pixel_format);
    if (driver.IsOpenGLES()) {
        const auto& downloader_es = runtime.GetDownloaderES();
        downloader_es.GetTexImage(GL_TEXTURE_2D, 0, tuple.format, tuple.type,
                                  rect_height, rect_width, 0);
    } else {
        glGetTexImage(GL_TEXTURE_2D, 0, tuple.format, tuple.type, 0);
    }
}

void Surface::ScaledUpload(const VideoCore::BufferTextureCopy& upload) {
    const u32 rect_width = upload.texture_rect.GetWidth();
    const u32 rect_height = upload.texture_rect.GetHeight();

    OGLTexture unscaled_tex = runtime.Allocate(rect_width, rect_height, pixel_format,
                                               VideoCore::TextureType::Texture2D);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, unscaled_tex.handle);

    const auto& tuple = runtime.GetFormatTuple(pixel_format);
    glTexSubImage2D(GL_TEXTURE_2D, upload.texture_level, 0, 0, rect_width, rect_height,
                    tuple.format, tuple.type, 0);

    const auto scaled_rect = upload.texture_rect * res_scale;
    const auto unscaled_rect = VideoCore::Rect2D{0, rect_height, rect_width, 0};
    const auto& filterer = runtime.GetFilterer();
    if (!filterer.Filter(unscaled_tex, unscaled_rect, texture, scaled_rect, type)) {
        runtime.BindFramebuffer(GL_READ_FRAMEBUFFER, 0, GL_TEXTURE_2D, type, unscaled_tex);
        runtime.BindFramebuffer(GL_DRAW_FRAMEBUFFER, upload.texture_level, GL_TEXTURE_2D, type, texture);

        // If filtering fails, resort to normal blitting
        glBlitFramebuffer(0, 0, rect_width, rect_height,
                          upload.texture_rect.left, upload.texture_rect.bottom,
                          upload.texture_rect.right, upload.texture_rect.top,
                          MakeBufferMask(type), GL_LINEAR);
    }
}

} // namespace OpenGL
