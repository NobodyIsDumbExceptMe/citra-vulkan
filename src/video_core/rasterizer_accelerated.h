// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once
#include "video_core/rasterizer_interface.h"

namespace VideoCore {

class RasterizerAccelerated : public RasterizerInterface {
public:
    virtual ~RasterizerAccelerated() = default;

    void UpdatePagesCachedCount(PAddr addr, u32 size, int delta) override;

    void ClearAll(bool flush) override;

private:
    std::array<u16, 0x30000> cached_pages{};
};
} // namespace VideoCore
