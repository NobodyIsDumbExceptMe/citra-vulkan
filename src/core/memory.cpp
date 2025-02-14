// Copyright 2015 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <cstring>
#include <boost/serialization/array.hpp>
#include <boost/serialization/binary_object.hpp>
#include "audio_core/dsp_interface.h"
#include "common/archives.h"
#include "common/assert.h"
#include "common/common_types.h"
#include "common/logging/log.h"
#include "common/swap.h"
#include "core/arm/arm_interface.h"
#include "core/core.h"
#include "core/global.h"
#include "core/hle/kernel/process.h"
#include "core/memory.h"
#include "core/settings.h"
#include "video_core/renderer_base.h"
#include "video_core/video_core.h"

SERIALIZE_EXPORT_IMPL(Memory::MemorySystem::BackingMemImpl<Memory::Region::FCRAM>)
SERIALIZE_EXPORT_IMPL(Memory::MemorySystem::BackingMemImpl<Memory::Region::VRAM>)
SERIALIZE_EXPORT_IMPL(Memory::MemorySystem::BackingMemImpl<Memory::Region::DSP>)
SERIALIZE_EXPORT_IMPL(Memory::MemorySystem::BackingMemImpl<Memory::Region::N3DS>)

namespace Memory {

void PageTable::Clear() {
    pointers.raw.fill(nullptr);
    pointers.refs.fill(MemoryRef());
    attributes.fill(PageType::Unmapped);
}

class RasterizerCacheMarker {
public:
    void Mark(VAddr addr, bool cached) {
        bool* p = At(addr);
        if (p)
            *p = cached;
    }

    bool IsCached(VAddr addr) {
        bool* p = At(addr);
        if (p)
            return *p;
        return false;
    }

private:
    bool* At(VAddr addr) {
        if (addr >= VRAM_VADDR && addr < VRAM_VADDR_END) {
            return &vram[(addr - VRAM_VADDR) / CITRA_PAGE_SIZE];
        }
        if (addr >= LINEAR_HEAP_VADDR && addr < LINEAR_HEAP_VADDR_END) {
            return &linear_heap[(addr - LINEAR_HEAP_VADDR) / CITRA_PAGE_SIZE];
        }
        if (addr >= NEW_LINEAR_HEAP_VADDR && addr < NEW_LINEAR_HEAP_VADDR_END) {
            return &new_linear_heap[(addr - NEW_LINEAR_HEAP_VADDR) / CITRA_PAGE_SIZE];
        }
        return nullptr;
    }

    std::array<bool, VRAM_SIZE / CITRA_PAGE_SIZE> vram{};
    std::array<bool, LINEAR_HEAP_SIZE / CITRA_PAGE_SIZE> linear_heap{};
    std::array<bool, NEW_LINEAR_HEAP_SIZE / CITRA_PAGE_SIZE> new_linear_heap{};

    static_assert(sizeof(bool) == 1);
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int file_version) {
        ar& vram;
        ar& linear_heap;
        ar& new_linear_heap;
    }
};

class MemorySystem::Impl {
public:
    // Visual Studio would try to allocate these on compile time if they are std::array, which would
    // exceed the memory limit.
    std::unique_ptr<u8[]> fcram = std::make_unique<u8[]>(Memory::FCRAM_N3DS_SIZE);
    std::unique_ptr<u8[]> vram = std::make_unique<u8[]>(Memory::VRAM_SIZE);
    std::unique_ptr<u8[]> n3ds_extra_ram = std::make_unique<u8[]>(Memory::N3DS_EXTRA_RAM_SIZE);

    std::shared_ptr<PageTable> current_page_table = nullptr;
    RasterizerCacheMarker cache_marker;
    std::vector<std::shared_ptr<PageTable>> page_table_list;

    AudioCore::DspInterface* dsp = nullptr;

    std::shared_ptr<BackingMem> fcram_mem;
    std::shared_ptr<BackingMem> vram_mem;
    std::shared_ptr<BackingMem> n3ds_extra_ram_mem;
    std::shared_ptr<BackingMem> dsp_mem;

    Impl();

    const u8* GetPtr(Region r) const {
        switch (r) {
        case Region::VRAM:
            return vram.get();
        case Region::DSP:
            return dsp->GetDspMemory().data();
        case Region::FCRAM:
            return fcram.get();
        case Region::N3DS:
            return n3ds_extra_ram.get();
        default:
            UNREACHABLE();
        }
    }

    u8* GetPtr(Region r) {
        switch (r) {
        case Region::VRAM:
            return vram.get();
        case Region::DSP:
            return dsp->GetDspMemory().data();
        case Region::FCRAM:
            return fcram.get();
        case Region::N3DS:
            return n3ds_extra_ram.get();
        default:
            UNREACHABLE();
        }
    }

    u32 GetSize(Region r) const {
        switch (r) {
        case Region::VRAM:
            return VRAM_SIZE;
        case Region::DSP:
            return DSP_RAM_SIZE;
        case Region::FCRAM:
            return FCRAM_N3DS_SIZE;
        case Region::N3DS:
            return N3DS_EXTRA_RAM_SIZE;
        default:
            UNREACHABLE();
        }
    }

    void WalkBlock(const Kernel::Process& process, const VAddr src_addr, const std::size_t size,
                   auto on_unmapped, auto on_memory, auto on_special, auto on_rasterizer, auto increment) {
        auto& page_table = *process.vm_manager.page_table;

        std::size_t remaining_size = size;
        std::size_t page_index = src_addr >> CITRA_PAGE_BITS;
        std::size_t page_offset = src_addr & CITRA_PAGE_MASK;

        while (remaining_size > 0) {
            const std::size_t copy_amount = std::min(CITRA_PAGE_SIZE - page_offset, remaining_size);
            const VAddr current_vaddr = static_cast<VAddr>((page_index << CITRA_PAGE_BITS) + page_offset);

            switch (page_table.attributes[page_index]) {
            case PageType::Unmapped: {
                on_unmapped(copy_amount, current_vaddr);
                break;
            }
            case PageType::Memory: {
                DEBUG_ASSERT(page_table.pointers[page_index]);
                u8* const src_ptr = page_table.pointers[page_index] + page_offset;
                on_memory(copy_amount, src_ptr);
                break;
            }
            case PageType::Special: {
                MMIORegionPointer handler = GetMMIOHandler(page_table, current_vaddr);
                DEBUG_ASSERT(handler);
                on_special(handler, copy_amount, current_vaddr);
                break;
            }
            case PageType::RasterizerCachedMemory: {
                u8* const rasterizer_ptr = GetPointerForRasterizerCache(current_vaddr);
                on_rasterizer(current_vaddr, copy_amount, rasterizer_ptr);
                break;
            }
            default:
                UNREACHABLE();
            }

            page_index++;
            page_offset = 0;
            increment(copy_amount);
            remaining_size -= copy_amount;
        }
    }

    template <bool UNSAFE>
    void ReadBlockImpl(const Kernel::Process& process, const VAddr src_addr, void* dest_buffer,
                       const std::size_t size) {
        WalkBlock(
            process, src_addr, size,
            [src_addr, size, &dest_buffer](const std::size_t copy_amount,
                                           const VAddr current_vaddr) {
                LOG_ERROR(HW_Memory,
                          "Unmapped ReadBlock @ 0x{:016X} (start address = 0x{:016X}, size = {})",
                          current_vaddr, src_addr, size);
                std::memset(dest_buffer, 0, copy_amount);
            },
            [&dest_buffer](const std::size_t copy_amount, const u8* const src_ptr) {
                std::memcpy(dest_buffer, src_ptr, copy_amount);
            },
            [&dest_buffer](MMIORegionPointer& handler,
                           const std::size_t copy_amount,
                           const VAddr current_vaddr) {
                handler->ReadBlock(current_vaddr, dest_buffer, copy_amount);
            },
            [&dest_buffer](const VAddr current_vaddr,
                           const std::size_t copy_amount,
                           const u8* const rasterizer_ptr) {
                if constexpr (!UNSAFE) {
                    RasterizerFlushVirtualRegion(current_vaddr, static_cast<u32>(copy_amount),
                                                 FlushMode::Flush);
                }
                std::memcpy(dest_buffer, rasterizer_ptr, copy_amount);
            },
            [&dest_buffer](const std::size_t copy_amount) {
                dest_buffer = static_cast<u8*>(dest_buffer) + copy_amount;
            });
    }

    template <bool UNSAFE>
    void WriteBlockImpl(const Kernel::Process& process, const VAddr dest_addr,
                        const void* src_buffer, const std::size_t size) {
        WalkBlock(
            process, dest_addr, size,
            [dest_addr, size](const std::size_t copy_amount, const VAddr current_vaddr) {
                LOG_ERROR(HW_Memory,
                          "Unmapped WriteBlock @ 0x{:016X} (start address = 0x{:016X}, size = {})",
                          current_vaddr, dest_addr, size);
            },
            [&src_buffer](const std::size_t copy_amount, u8* const dest_ptr) {
                std::memcpy(dest_ptr, src_buffer, copy_amount);
            },
            [&src_buffer](MMIORegionPointer& handler,
                           const std::size_t copy_amount,
                           const VAddr current_vaddr) {
                handler->WriteBlock(current_vaddr, src_buffer, copy_amount);
            },
            [&src_buffer](const VAddr current_vaddr,
                          const std::size_t copy_amount, u8* const host_ptr) {
                if constexpr (!UNSAFE) {
                    RasterizerFlushVirtualRegion(current_vaddr, static_cast<u32>(copy_amount),
                                                 FlushMode::Invalidate);
                }
                std::memcpy(host_ptr, src_buffer, copy_amount);
            },
            [&src_buffer](const std::size_t copy_amount) {
                src_buffer = static_cast<const u8*>(src_buffer) + copy_amount;
            });
    }

    MemoryRef GetPointerForRasterizerCache(VAddr addr) const {
        if (addr >= LINEAR_HEAP_VADDR && addr < LINEAR_HEAP_VADDR_END) {
            return {fcram_mem, addr - LINEAR_HEAP_VADDR};
        }
        if (addr >= NEW_LINEAR_HEAP_VADDR && addr < NEW_LINEAR_HEAP_VADDR_END) {
            return {fcram_mem, addr - NEW_LINEAR_HEAP_VADDR};
        }
        if (addr >= VRAM_VADDR && addr < VRAM_VADDR_END) {
            return {vram_mem, addr - VRAM_VADDR};
        }

        UNREACHABLE();
        return MemoryRef{};
    }

    /**
     * This function should only be called for virtual addreses with attribute `PageType::Special`.
     */
    MMIORegionPointer GetMMIOHandler(const PageTable& page_table, VAddr vaddr) {
        for (const auto& region : page_table.special_regions) {
            if (vaddr >= region.base && vaddr < (region.base + region.size)) {
                return region.handler;
            }
        }

        ASSERT_MSG(false, "Mapped IO page without a handler @ {:08X}", vaddr);
        return nullptr; // Should never happen
    }

private:
    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int file_version) {
        bool save_n3ds_ram = Settings::values.is_new_3ds;
        ar& save_n3ds_ram;
        ar& boost::serialization::make_binary_object(vram.get(), Memory::VRAM_SIZE);
        ar& boost::serialization::make_binary_object(
            fcram.get(), save_n3ds_ram ? Memory::FCRAM_N3DS_SIZE : Memory::FCRAM_SIZE);
        ar& boost::serialization::make_binary_object(
            n3ds_extra_ram.get(), save_n3ds_ram ? Memory::N3DS_EXTRA_RAM_SIZE : 0);
        ar& cache_marker;
        ar& page_table_list;
        // dsp is set from Core::System at startup
        ar& current_page_table;
        ar& fcram_mem;
        ar& vram_mem;
        ar& n3ds_extra_ram_mem;
        ar& dsp_mem;
    }
};

// We use this rather than BufferMem because we don't want new objects to be allocated when
// deserializing. This avoids unnecessary memory thrashing.
template <Region R>
class MemorySystem::BackingMemImpl : public BackingMem {
public:
    BackingMemImpl() : impl(*Core::Global<Core::System>().Memory().impl) {}
    explicit BackingMemImpl(MemorySystem::Impl& impl_) : impl(impl_) {}
    u8* GetPtr() override {
        return impl.GetPtr(R);
    }
    const u8* GetPtr() const override {
        return impl.GetPtr(R);
    }
    std::size_t GetSize() const override {
        return impl.GetSize(R);
    }

private:
    MemorySystem::Impl& impl;

    template <class Archive>
    void serialize(Archive& ar, const unsigned int) {
        ar& boost::serialization::base_object<BackingMem>(*this);
    }
    friend class boost::serialization::access;
};

MemorySystem::Impl::Impl()
    : fcram_mem(std::make_shared<BackingMemImpl<Region::FCRAM>>(*this)),
      vram_mem(std::make_shared<BackingMemImpl<Region::VRAM>>(*this)),
      n3ds_extra_ram_mem(std::make_shared<BackingMemImpl<Region::N3DS>>(*this)),
      dsp_mem(std::make_shared<BackingMemImpl<Region::DSP>>(*this)) {}

MemorySystem::MemorySystem() : impl(std::make_unique<Impl>()) {}
MemorySystem::~MemorySystem() = default;

template <class Archive>
void MemorySystem::serialize(Archive& ar, const unsigned int file_version) {
    ar&* impl.get();
}

SERIALIZE_IMPL(MemorySystem)

void MemorySystem::SetCurrentPageTable(std::shared_ptr<PageTable> page_table) {
    impl->current_page_table = page_table;
}

std::shared_ptr<PageTable> MemorySystem::GetCurrentPageTable() const {
    return impl->current_page_table;
}

void MemorySystem::MapPages(PageTable& page_table, u32 base, u32 size, MemoryRef memory,
                            PageType type) {
    LOG_DEBUG(HW_Memory, "Mapping {} onto {:08X}-{:08X}", (void*)memory.GetPtr(), base * CITRA_PAGE_SIZE,
              (base + size) * CITRA_PAGE_SIZE);

    RasterizerFlushVirtualRegion(base << CITRA_PAGE_BITS, size * CITRA_PAGE_SIZE,
                                 FlushMode::FlushAndInvalidate);

    u32 end = base + size;
    while (base != end) {
        ASSERT_MSG(base < PAGE_TABLE_NUM_ENTRIES, "out of range mapping at {:08X}", base);

        page_table.attributes[base] = type;
        page_table.pointers[base] = memory;

        // If the memory to map is already rasterizer-cached, mark the page
        if (type == PageType::Memory && impl->cache_marker.IsCached(base * CITRA_PAGE_SIZE)) {
            page_table.attributes[base] = PageType::RasterizerCachedMemory;
            page_table.pointers[base] = nullptr;
        }

        base += 1;
        if (memory != nullptr && memory.GetSize() > CITRA_PAGE_SIZE)
            memory += CITRA_PAGE_SIZE;
    }
}

void MemorySystem::MapMemoryRegion(PageTable& page_table, VAddr base, u32 size, MemoryRef target) {
    ASSERT_MSG((size & CITRA_PAGE_MASK) == 0, "non-page aligned size: {:08X}", size);
    ASSERT_MSG((base & CITRA_PAGE_MASK) == 0, "non-page aligned base: {:08X}", base);
    MapPages(page_table, base / CITRA_PAGE_SIZE, size / CITRA_PAGE_SIZE, target, PageType::Memory);
}

void MemorySystem::MapIoRegion(PageTable& page_table, VAddr base, u32 size,
                               MMIORegionPointer mmio_handler) {
    ASSERT_MSG((size & CITRA_PAGE_MASK) == 0, "non-page aligned size: {:08X}", size);
    ASSERT_MSG((base & CITRA_PAGE_MASK) == 0, "non-page aligned base: {:08X}", base);
    MapPages(page_table, base / CITRA_PAGE_SIZE, size / CITRA_PAGE_SIZE, nullptr, PageType::Special);

    page_table.special_regions.emplace_back(SpecialRegion{base, size, mmio_handler});
}

void MemorySystem::UnmapRegion(PageTable& page_table, VAddr base, u32 size) {
    ASSERT_MSG((size & CITRA_PAGE_MASK) == 0, "non-page aligned size: {:08X}", size);
    ASSERT_MSG((base & CITRA_PAGE_MASK) == 0, "non-page aligned base: {:08X}", base);
    MapPages(page_table, base / CITRA_PAGE_SIZE, size / CITRA_PAGE_SIZE, nullptr, PageType::Unmapped);
}

MemoryRef MemorySystem::GetPointerForRasterizerCache(VAddr addr) const {
    return impl->GetPointerForRasterizerCache(addr);
}

void MemorySystem::RegisterPageTable(std::shared_ptr<PageTable> page_table) {
    impl->page_table_list.push_back(page_table);
}

void MemorySystem::UnregisterPageTable(std::shared_ptr<PageTable> page_table) {
    auto it = std::find(impl->page_table_list.begin(), impl->page_table_list.end(), page_table);
    if (it != impl->page_table_list.end()) {
        impl->page_table_list.erase(it);
    }
}

template <typename T>
T ReadMMIO(MMIORegionPointer mmio_handler, VAddr addr);

template <typename T>
T MemorySystem::Read(const VAddr vaddr) {
    const u8* page_pointer = impl->current_page_table->pointers[vaddr >> CITRA_PAGE_BITS];
    if (page_pointer) {
        // NOTE: Avoid adding any extra logic to this fast-path block
        T value;
        std::memcpy(&value, &page_pointer[vaddr & CITRA_PAGE_MASK], sizeof(T));
        return value;
    }

    PageType type = impl->current_page_table->attributes[vaddr >> CITRA_PAGE_BITS];
    switch (type) {
    case PageType::Unmapped:
        LOG_ERROR(HW_Memory, "unmapped Read{} @ 0x{:08X} at PC 0x{:08X}", sizeof(T) * 8, vaddr,
                  Core::GetRunningCore().GetPC());
        return 0;
    case PageType::Memory:
        ASSERT_MSG(false, "Mapped memory page without a pointer @ {:08X}", vaddr);
        break;
    case PageType::RasterizerCachedMemory: {
        RasterizerFlushVirtualRegion(vaddr, sizeof(T), FlushMode::Flush);

        T value;
        std::memcpy(&value, GetPointerForRasterizerCache(vaddr), sizeof(T));
        return value;
    }
    case PageType::Special:
        return ReadMMIO<T>(impl->GetMMIOHandler(*impl->current_page_table, vaddr), vaddr);
    default:
        UNREACHABLE();
    }

    return T{};
}

template <typename T>
void WriteMMIO(MMIORegionPointer mmio_handler, VAddr addr, const T data);

template <typename T>
void MemorySystem::Write(const VAddr vaddr, const T data) {
    u8* page_pointer = impl->current_page_table->pointers[vaddr >> CITRA_PAGE_BITS];
    if (page_pointer) {
        // NOTE: Avoid adding any extra logic to this fast-path block
        std::memcpy(&page_pointer[vaddr & CITRA_PAGE_MASK], &data, sizeof(T));
        return;
    }

    PageType type = impl->current_page_table->attributes[vaddr >> CITRA_PAGE_BITS];
    switch (type) {
    case PageType::Unmapped:
        LOG_ERROR(HW_Memory, "unmapped Write{} 0x{:08X} @ 0x{:08X} at PC 0x{:08X}",
                  sizeof(data) * 8, (u32)data, vaddr, Core::GetRunningCore().GetPC());
        return;
    case PageType::Memory:
        ASSERT_MSG(false, "Mapped memory page without a pointer @ {:08X}", vaddr);
        break;
    case PageType::RasterizerCachedMemory: {
        RasterizerFlushVirtualRegion(vaddr, sizeof(T), FlushMode::Invalidate);
        std::memcpy(GetPointerForRasterizerCache(vaddr), &data, sizeof(T));
        break;
    }
    case PageType::Special:
        WriteMMIO<T>(impl->GetMMIOHandler(*impl->current_page_table, vaddr), vaddr, data);
        break;
    default:
        UNREACHABLE();
    }
}

bool MemorySystem::IsValidVirtualAddress(const Kernel::Process& process, const VAddr vaddr) {
    auto& page_table = *process.vm_manager.page_table;

    auto page_pointer = page_table.pointers[vaddr >> CITRA_PAGE_BITS];
    if (page_pointer)
        return true;

    if (page_table.attributes[vaddr >> CITRA_PAGE_BITS] == PageType::RasterizerCachedMemory)
        return true;

    if (page_table.attributes[vaddr >> CITRA_PAGE_BITS] != PageType::Special)
        return false;

    MMIORegionPointer mmio_region = impl->GetMMIOHandler(page_table, vaddr);
    if (mmio_region) {
        return mmio_region->IsValidAddress(vaddr);
    }

    return false;
}

bool MemorySystem::IsValidPhysicalAddress(const PAddr paddr) const {
    return GetPhysicalRef(paddr);
}

u8* MemorySystem::GetPointer(const VAddr vaddr) {
    u8* page_pointer = impl->current_page_table->pointers[vaddr >> CITRA_PAGE_BITS];
    if (page_pointer) {
        return page_pointer + (vaddr & CITRA_PAGE_MASK);
    }

    if (impl->current_page_table->attributes[vaddr >> CITRA_PAGE_BITS] ==
        PageType::RasterizerCachedMemory) {
        return GetPointerForRasterizerCache(vaddr);
    }

    LOG_ERROR(HW_Memory, "unknown GetPointer @ 0x{:08x} at PC 0x{:08X}", vaddr,
              Core::GetRunningCore().GetPC());
    return nullptr;
}

const u8* MemorySystem::GetPointer(const VAddr vaddr) const {
    const u8* page_pointer = impl->current_page_table->pointers[vaddr >> CITRA_PAGE_BITS];
    if (page_pointer) {
        return page_pointer + (vaddr & CITRA_PAGE_MASK);
    }

    if (impl->current_page_table->attributes[vaddr >> CITRA_PAGE_BITS] ==
        PageType::RasterizerCachedMemory) {
        return GetPointerForRasterizerCache(vaddr);
    }

    LOG_ERROR(HW_Memory, "unknown GetPointer @ 0x{:08x}", vaddr);
    return nullptr;
}

std::string MemorySystem::ReadCString(VAddr vaddr, std::size_t max_length) {
    std::string string;
    string.reserve(max_length);
    for (std::size_t i = 0; i < max_length; ++i) {
        char c = Read8(vaddr);
        if (c == '\0') {
            break;
        }

        string.push_back(c);
        ++vaddr;
    }

    string.shrink_to_fit();
    return string;
}

u8* MemorySystem::GetPhysicalPointer(PAddr address) {
    return GetPhysicalRef(address);
}

MemoryRef MemorySystem::GetPhysicalRef(PAddr address) const {
    constexpr std::array memory_areas = {
        std::make_pair(VRAM_PADDR, VRAM_SIZE),
        std::make_pair(DSP_RAM_PADDR, DSP_RAM_SIZE),
        std::make_pair(FCRAM_PADDR, FCRAM_N3DS_SIZE),
        std::make_pair(N3DS_EXTRA_RAM_PADDR, N3DS_EXTRA_RAM_SIZE),
    };

    const auto area =
        std::ranges::find_if(memory_areas, [&](const auto& area) {
            // Note: the region end check is inclusive because the user can pass in an address that
            // represents an open right bound
            return address >= area.first && address <= area.first + area.second;
        });

    if (area == memory_areas.end()) {
        LOG_ERROR(HW_Memory, "Unknown GetPhysicalPointer @ {:#08X} at PC {:#08X}", address,
                  Core::GetRunningCore().GetPC());
        return nullptr;
    }

    u32 offset_into_region = address - area->first;

    std::shared_ptr<BackingMem> target_mem = nullptr;
    switch (area->first) {
    case VRAM_PADDR:
        target_mem = impl->vram_mem;
        break;
    case DSP_RAM_PADDR:
        target_mem = impl->dsp_mem;
        break;
    case FCRAM_PADDR:
        target_mem = impl->fcram_mem;
        break;
    case N3DS_EXTRA_RAM_PADDR:
        target_mem = impl->n3ds_extra_ram_mem;
        break;
    default:
        UNREACHABLE();
    }
    if (offset_into_region >= target_mem->GetSize()) {
        return {nullptr};
    }

    return {target_mem, offset_into_region};
}

/// For a rasterizer-accessible PAddr, gets a list of all possible VAddr
static std::vector<VAddr> PhysicalToVirtualAddressForRasterizer(PAddr addr) {
    if (addr >= VRAM_PADDR && addr < VRAM_PADDR_END) {
        return {addr - VRAM_PADDR + VRAM_VADDR};
    }
    if (addr >= FCRAM_PADDR && addr < FCRAM_PADDR_END) {
        return {addr - FCRAM_PADDR + LINEAR_HEAP_VADDR, addr - FCRAM_PADDR + NEW_LINEAR_HEAP_VADDR};
    }
    if (addr >= FCRAM_PADDR_END && addr < FCRAM_N3DS_PADDR_END) {
        return {addr - FCRAM_PADDR + NEW_LINEAR_HEAP_VADDR};
    }
    // While the physical <-> virtual mapping is 1:1 for the regions supported by the cache,
    // some games (like Pokemon Super Mystery Dungeon) will try to use textures that go beyond
    // the end address of VRAM, causing the Virtual->Physical translation to fail when flushing
    // parts of the texture.
    LOG_ERROR(HW_Memory,
              "Trying to use invalid physical address for rasterizer: {:08X} at PC 0x{:08X}", addr,
              Core::GetRunningCore().GetPC());
    return {};
}

void MemorySystem::RasterizerMarkRegionCached(PAddr start, u32 size, bool cached) {
    if (start == 0) {
        return;
    }

    u32 num_pages = ((start + size - 1) >> CITRA_PAGE_BITS) - (start >> CITRA_PAGE_BITS) + 1;
    PAddr paddr = start;

    for (unsigned i = 0; i < num_pages; ++i, paddr += CITRA_PAGE_SIZE) {
        for (VAddr vaddr : PhysicalToVirtualAddressForRasterizer(paddr)) {
            impl->cache_marker.Mark(vaddr, cached);
            for (auto page_table : impl->page_table_list) {
                PageType& page_type = page_table->attributes[vaddr >> CITRA_PAGE_BITS];

                if (cached) {
                    // Switch page type to cached if now cached
                    switch (page_type) {
                    case PageType::Unmapped:
                        // It is not necessary for a process to have this region mapped into its
                        // address space, for example, a system module need not have a VRAM mapping.
                        break;
                    case PageType::Memory:
                        page_type = PageType::RasterizerCachedMemory;
                        page_table->pointers[vaddr >> CITRA_PAGE_BITS] = nullptr;
                        break;
                    default:
                        UNREACHABLE();
                    }
                } else {
                    // Switch page type to uncached if now uncached
                    switch (page_type) {
                    case PageType::Unmapped:
                        // It is not necessary for a process to have this region mapped into its
                        // address space, for example, a system module need not have a VRAM mapping.
                        break;
                    case PageType::RasterizerCachedMemory: {
                        page_type = PageType::Memory;
                        page_table->pointers[vaddr >> CITRA_PAGE_BITS] =
                            GetPointerForRasterizerCache(vaddr & ~CITRA_PAGE_MASK);
                        break;
                    }
                    default:
                        UNREACHABLE();
                    }
                }
            }
        }
    }
}

void RasterizerFlushRegion(PAddr start, u32 size) {
    if (VideoCore::g_renderer == nullptr) {
        return;
    }

    VideoCore::g_renderer->Rasterizer()->FlushRegion(start, size);
}

void RasterizerInvalidateRegion(PAddr start, u32 size) {
    if (VideoCore::g_renderer == nullptr) {
        return;
    }

    VideoCore::g_renderer->Rasterizer()->InvalidateRegion(start, size);
}

void RasterizerFlushAndInvalidateRegion(PAddr start, u32 size) {
    // Since pages are unmapped on shutdown after video core is shutdown, the renderer may be
    // null here
    if (VideoCore::g_renderer == nullptr) {
        return;
    }

    VideoCore::g_renderer->Rasterizer()->FlushAndInvalidateRegion(start, size);
}

void RasterizerClearAll(bool flush) {
    // Since pages are unmapped on shutdown after video core is shutdown, the renderer may be
    // null here
    if (VideoCore::g_renderer == nullptr) {
        return;
    }

    VideoCore::g_renderer->Rasterizer()->ClearAll(flush);
}

void RasterizerFlushVirtualRegion(VAddr start, u32 size, FlushMode mode) {
    // Since pages are unmapped on shutdown after video core is shutdown, the renderer may be
    // null here
    if (VideoCore::g_renderer == nullptr) {
        return;
    }

    VAddr end = start + size;

    auto CheckRegion = [&](VAddr region_start, VAddr region_end, PAddr paddr_region_start) {
        if (start >= region_end || end <= region_start) {
            // No overlap with region
            return;
        }

        VAddr overlap_start = std::max(start, region_start);
        VAddr overlap_end = std::min(end, region_end);
        PAddr physical_start = paddr_region_start + (overlap_start - region_start);
        u32 overlap_size = overlap_end - overlap_start;

        auto* rasterizer = VideoCore::g_renderer->Rasterizer();
        switch (mode) {
        case FlushMode::Flush:
            rasterizer->FlushRegion(physical_start, overlap_size);
            break;
        case FlushMode::Invalidate:
            rasterizer->InvalidateRegion(physical_start, overlap_size);
            break;
        case FlushMode::FlushAndInvalidate:
            rasterizer->FlushAndInvalidateRegion(physical_start, overlap_size);
            break;
        }
    };

    CheckRegion(LINEAR_HEAP_VADDR, LINEAR_HEAP_VADDR_END, FCRAM_PADDR);
    CheckRegion(NEW_LINEAR_HEAP_VADDR, NEW_LINEAR_HEAP_VADDR_END, FCRAM_PADDR);
    CheckRegion(VRAM_VADDR, VRAM_VADDR_END, VRAM_PADDR);
}

u8 MemorySystem::Read8(const VAddr addr) {
    return Read<u8>(addr);
}

u16 MemorySystem::Read16(const VAddr addr) {
    return Read<u16_le>(addr);
}

u32 MemorySystem::Read32(const VAddr addr) {
    return Read<u32_le>(addr);
}

u64 MemorySystem::Read64(const VAddr addr) {
    return Read<u64_le>(addr);
}

void MemorySystem::ReadBlock(const Kernel::Process& process, const VAddr src_addr,
                             void* dest_buffer, const std::size_t size) {
    return impl->ReadBlockImpl<false>(process, src_addr, dest_buffer, size);
}

void MemorySystem::ReadBlock(VAddr src_addr, void* dest_buffer, std::size_t size) {
    const auto& process = *Core::System::GetInstance().Kernel().GetCurrentProcess();
    return impl->ReadBlockImpl<false>(process, src_addr, dest_buffer, size);
}

void MemorySystem::Write8(const VAddr addr, const u8 data) {
    Write<u8>(addr, data);
}

void MemorySystem::Write16(const VAddr addr, const u16 data) {
    Write<u16_le>(addr, data);
}

void MemorySystem::Write32(const VAddr addr, const u32 data) {
    Write<u32_le>(addr, data);
}

void MemorySystem::Write64(const VAddr addr, const u64 data) {
    Write<u64_le>(addr, data);
}

void MemorySystem::WriteBlock(const Kernel::Process& process, const VAddr dest_addr,
                              const void* src_buffer, const std::size_t size) {
    return impl->WriteBlockImpl<false>(process, dest_addr, src_buffer, size);
}

void MemorySystem::WriteBlock(const VAddr dest_addr, const void* src_buffer, const std::size_t size) {
    auto& process = *Core::System::GetInstance().Kernel().GetCurrentProcess();
    return impl->WriteBlockImpl<false>(process, dest_addr, src_buffer, size);
}

void MemorySystem::ZeroBlock(const Kernel::Process& process, const VAddr dest_addr,
                             const std::size_t size) {
    static const std::array<u8, CITRA_PAGE_SIZE> zeros{0};

    impl->WalkBlock(
        process, dest_addr, size,
        [dest_addr, size](const std::size_t copy_amount, const VAddr current_vaddr) {
            LOG_ERROR(HW_Memory,
                      "Unmapped ZeroBlock @ 0x{:016X} (start address = 0x{:016X}, size = {})",
                      current_vaddr, dest_addr, size);
        },
        [](const std::size_t copy_amount, u8* const dest_ptr) {
            std::memset(dest_ptr, 0, copy_amount);
        },
        [&zeros = zeros](MMIORegionPointer& handler,
                         const std::size_t copy_amount,
                         const VAddr current_vaddr) {
            handler->WriteBlock(current_vaddr, zeros.data(), copy_amount);
        },
        [](const VAddr current_vaddr, const std::size_t copy_amount,
                           u8* const rasterizer_ptr) {
            RasterizerFlushVirtualRegion(current_vaddr, static_cast<u32>(copy_amount),
                                         FlushMode::Invalidate);
            std::memset(rasterizer_ptr, 0, copy_amount);
        },
        [](const std::size_t copy_amount) {});
}

void MemorySystem::CopyBlock(const Kernel::Process& process, VAddr dest_addr, VAddr src_addr,
                             const std::size_t size) {
    CopyBlock(process, process, dest_addr, src_addr, size);
}

void MemorySystem::CopyBlock(const Kernel::Process& dest_process,
                             const Kernel::Process& src_process, VAddr dest_addr, VAddr src_addr,
                             std::size_t size) {
    std::array<u8, CITRA_PAGE_SIZE> copy_buffer{};

    impl->WalkBlock(
        src_process, src_addr, size,
        [this, &dest_process, &dest_addr, &src_addr, size](const std::size_t copy_amount,
                                                           const VAddr current_vaddr) {
            LOG_ERROR(HW_Memory,
                      "unmapped CopyBlock @ 0x{:08X} (start address = 0x{:08X}, size = {})",
                      current_vaddr, src_addr, size);
            ZeroBlock(dest_process, dest_addr, copy_amount);
        },
        [this, &dest_process, &dest_addr](const std::size_t copy_amount, const u8* const src_ptr) {
            impl->WriteBlockImpl<false>(dest_process, dest_addr, src_ptr, copy_amount);
        },
        [this, &dest_process, &dest_addr, &copy_buffer](MMIORegionPointer& handler,
                                                       const std::size_t copy_amount,
                                                       const VAddr current_vaddr) {
            handler->ReadBlock(current_vaddr, copy_buffer.data(), copy_amount);
            impl->WriteBlockImpl<false>(dest_process, dest_addr, copy_buffer.data(), copy_amount);
        },
        [this, &dest_process, &dest_addr](
            const VAddr current_vaddr, const std::size_t copy_amount, u8* const rasterizer_ptr) {
            RasterizerFlushVirtualRegion(current_vaddr, static_cast<u32>(copy_amount),
                                         FlushMode::Flush);
            impl->WriteBlockImpl<false>(dest_process, dest_addr, rasterizer_ptr, copy_amount);
        },
        [&dest_addr, &src_addr](const std::size_t copy_amount) {
            dest_addr += static_cast<VAddr>(copy_amount);
            src_addr += static_cast<VAddr>(copy_amount);
        });
}

template <>
u8 ReadMMIO<u8>(MMIORegionPointer mmio_handler, VAddr addr) {
    return mmio_handler->Read8(addr);
}

template <>
u16 ReadMMIO<u16>(MMIORegionPointer mmio_handler, VAddr addr) {
    return mmio_handler->Read16(addr);
}

template <>
u32 ReadMMIO<u32>(MMIORegionPointer mmio_handler, VAddr addr) {
    return mmio_handler->Read32(addr);
}

template <>
u64 ReadMMIO<u64>(MMIORegionPointer mmio_handler, VAddr addr) {
    return mmio_handler->Read64(addr);
}

template <>
void WriteMMIO<u8>(MMIORegionPointer mmio_handler, VAddr addr, const u8 data) {
    mmio_handler->Write8(addr, data);
}

template <>
void WriteMMIO<u16>(MMIORegionPointer mmio_handler, VAddr addr, const u16 data) {
    mmio_handler->Write16(addr, data);
}

template <>
void WriteMMIO<u32>(MMIORegionPointer mmio_handler, VAddr addr, const u32 data) {
    mmio_handler->Write32(addr, data);
}

template <>
void WriteMMIO<u64>(MMIORegionPointer mmio_handler, VAddr addr, const u64 data) {
    mmio_handler->Write64(addr, data);
}

u32 MemorySystem::GetFCRAMOffset(const u8* pointer) const {
    ASSERT(pointer >= impl->fcram.get() && pointer <= impl->fcram.get() + Memory::FCRAM_N3DS_SIZE);
    return static_cast<u32>(pointer - impl->fcram.get());
}

u8* MemorySystem::GetFCRAMPointer(std::size_t offset) {
    ASSERT(offset <= Memory::FCRAM_N3DS_SIZE);
    return impl->fcram.get() + offset;
}

const u8* MemorySystem::GetFCRAMPointer(std::size_t offset) const {
    ASSERT(offset <= Memory::FCRAM_N3DS_SIZE);
    return impl->fcram.get() + offset;
}

MemoryRef MemorySystem::GetFCRAMRef(std::size_t offset) const {
    ASSERT(offset <= Memory::FCRAM_N3DS_SIZE);
    return MemoryRef(impl->fcram_mem, offset);
}

void MemorySystem::SetDSP(AudioCore::DspInterface& dsp) {
    impl->dsp = &dsp;
}

} // namespace Memory
