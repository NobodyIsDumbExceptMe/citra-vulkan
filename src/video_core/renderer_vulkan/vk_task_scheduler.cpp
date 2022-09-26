// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#define VULKAN_HPP_NO_CONSTRUCTORS
#include "common/assert.h"
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_task_scheduler.h"
#include "video_core/renderer_vulkan/vk_instance.h"

namespace Vulkan {

TaskScheduler::TaskScheduler(const Instance& instance) : instance{instance} {
    vk::Device device = instance.GetDevice();
    const vk::CommandPoolCreateInfo command_pool_info = {
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = instance.GetGraphicsQueueFamilyIndex()
    };

    command_pool = device.createCommandPool(command_pool_info);

    // If supported, prefer timeline semaphores over binary ones
    if (instance.IsTimelineSemaphoreSupported()) {
        const vk::StructureChain timeline_info = {
            vk::SemaphoreCreateInfo{},
            vk::SemaphoreTypeCreateInfo{
                .semaphoreType = vk::SemaphoreType::eTimeline,
                .initialValue = 0
            }
        };

        timeline = device.createSemaphore(timeline_info.get());
    }

    constexpr std::array pool_sizes = {
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 1024},
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformBufferDynamic, 1024},
        vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 2048},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 2048},
        vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 2048},
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformTexelBuffer, 1024}
    };

    const vk::DescriptorPoolCreateInfo descriptor_pool_info = {
        .maxSets = 2048,
        .poolSizeCount = static_cast<u32>(pool_sizes.size()),
        .pPoolSizes = pool_sizes.data()
    };

    const vk::CommandBufferAllocateInfo buffer_info = {
        .commandPool = command_pool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 2 * SCHEDULER_COMMAND_COUNT
    };

    const auto command_buffers = device.allocateCommandBuffers(buffer_info);
    for (std::size_t i = 0; i < commands.size(); i++) {
        commands[i] = ExecutionSlot{
            .image_acquired = device.createSemaphore({}),
            .present_ready = device.createSemaphore({}),
            .fence = device.createFence({}),
            .descriptor_pool = device.createDescriptorPool(descriptor_pool_info),
            .render_command_buffer = command_buffers[2 * i],
            .upload_command_buffer = command_buffers[2 * i + 1],
        };
    }

    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
    };

    // Begin first command
    auto& command = commands[current_command];
    command.render_command_buffer.begin(begin_info);
    command.fence_counter = next_fence_counter++;
}

TaskScheduler::~TaskScheduler() {
    vk::Device device = instance.GetDevice();
    device.waitIdle();

    if (timeline) {
        device.destroySemaphore(timeline);
    }

    for (const auto& command : commands) {
        device.destroyFence(command.fence);
        device.destroySemaphore(command.image_acquired);
        device.destroySemaphore(command.present_ready);
        device.destroyDescriptorPool(command.descriptor_pool);
    }

    device.destroyCommandPool(command_pool);
}

void TaskScheduler::Synchronize(u32 slot) {
    const auto& command = commands[slot];
    vk::Device device = instance.GetDevice();

    u32 completed_counter = completed_fence_counter;
    if (instance.IsTimelineSemaphoreSupported()) {
        completed_counter = device.getSemaphoreCounterValue(timeline);
    }

    if (command.fence_counter > completed_counter) {
        if (instance.IsTimelineSemaphoreSupported()) {
            const vk::SemaphoreWaitInfo wait_info = {
                .semaphoreCount = 1,
                .pSemaphores = &timeline,
                .pValues = &command.fence_counter
            };

            if (device.waitSemaphores(wait_info, UINT64_MAX) != vk::Result::eSuccess) {
                LOG_ERROR(Render_Vulkan, "Waiting for fence counter {} failed!", command.fence_counter);
                UNREACHABLE();
            }

        } else if (device.waitForFences(command.fence, true, UINT64_MAX) != vk::Result::eSuccess) {
            LOG_ERROR(Render_Vulkan, "Waiting for fence counter {} failed!", command.fence_counter);
            UNREACHABLE();
        }

        completed_fence_counter = command.fence_counter;
    }

    device.resetFences(command.fence);
    device.resetDescriptorPool(command.descriptor_pool);
}

void TaskScheduler::WaitFence(u32 counter) {
    for (u32 i = 0; i < SCHEDULER_COMMAND_COUNT; i++) {
        if (commands[i].fence_counter == counter) {
            return Synchronize(i);
        }
    }

    LOG_CRITICAL(Render_Vulkan,"Invalid fence counter {}!", counter);
    UNREACHABLE();
}

void TaskScheduler::Submit(SubmitMode mode) {
    const auto& command = commands[current_command];
    command.render_command_buffer.end();
    if (command.use_upload_buffer) {
        command.upload_command_buffer.end();
    }

    u32 command_buffer_count = 0;
    std::array<vk::CommandBuffer, 2> command_buffers;

    if (command.use_upload_buffer) {
        command_buffers[command_buffer_count++] = command.upload_command_buffer;
    }

    command_buffers[command_buffer_count++] = command.render_command_buffer;

    const bool swapchain_sync = True(mode & SubmitMode::SwapchainSynced);
    if (instance.IsTimelineSemaphoreSupported()) {
        const u32 wait_semaphore_count = swapchain_sync ? 2u : 1u;
        const std::array wait_values{command.fence_counter - 1, 1ul};
        const std::array wait_semaphores{timeline, command.image_acquired};

        const u32 signal_semaphore_count = swapchain_sync ? 2u : 1u;
        const std::array signal_values{command.fence_counter, 0ul};
        const std::array signal_semaphores{timeline, command.present_ready};

        const vk::TimelineSemaphoreSubmitInfoKHR timeline_si = {
            .waitSemaphoreValueCount = wait_semaphore_count,
            .pWaitSemaphoreValues = wait_values.data(),
            .signalSemaphoreValueCount = signal_semaphore_count,
            .pSignalSemaphoreValues = signal_values.data()
        };

        const std::array<vk::PipelineStageFlags, 2> wait_stage_masks = {
            vk::PipelineStageFlagBits::eAllCommands,
            vk::PipelineStageFlagBits::eColorAttachmentOutput,
        };

        const vk::SubmitInfo submit_info = {
            .pNext = &timeline_si,
            .waitSemaphoreCount = wait_semaphore_count,
            .pWaitSemaphores = wait_semaphores.data(),
            .pWaitDstStageMask = wait_stage_masks.data(),
            .commandBufferCount = command_buffer_count,
            .pCommandBuffers = command_buffers.data(),
            .signalSemaphoreCount = signal_semaphore_count,
            .pSignalSemaphores = signal_semaphores.data(),
        };

        vk::Queue queue = instance.GetGraphicsQueue();
        queue.submit(submit_info);

    } else {
        const u32 signal_semaphore_count = swapchain_sync ? 1u : 0u;
        const u32 wait_semaphore_count = swapchain_sync ? 1u : 0u;
        const vk::PipelineStageFlags wait_stage_masks =
                vk::PipelineStageFlagBits::eColorAttachmentOutput;

        const vk::SubmitInfo submit_info = {
            .waitSemaphoreCount = wait_semaphore_count,
            .pWaitSemaphores = &command.image_acquired,
            .pWaitDstStageMask = &wait_stage_masks,
            .commandBufferCount = command_buffer_count,
            .pCommandBuffers = command_buffers.data(),
            .signalSemaphoreCount = signal_semaphore_count,
            .pSignalSemaphores = &command.present_ready,
        };

        vk::Queue queue = instance.GetGraphicsQueue();
        queue.submit(submit_info, command.fence);
    }

    // Block host until the GPU catches up
    if (True(mode & SubmitMode::Flush)) {
        Synchronize(current_command);
    }

    // Switch to next cmdbuffer.
    if (False(mode & SubmitMode::Shutdown)) {
        SwitchSlot();
    }
}

u64 TaskScheduler::GetFenceCounter() const {
    vk::Device device = instance.GetDevice();
    if (instance.IsTimelineSemaphoreSupported()) {
        return device.getSemaphoreCounterValue(timeline);
    }

    return completed_fence_counter;
}

vk::CommandBuffer TaskScheduler::GetUploadCommandBuffer() {
    auto& command = commands[current_command];
    if (!command.use_upload_buffer) {
        const vk::CommandBufferBeginInfo begin_info = {
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
        };

        command.upload_command_buffer.begin(begin_info);
        command.use_upload_buffer = true;
    }

    return command.upload_command_buffer;
}

void TaskScheduler::SwitchSlot() {
    current_command = (current_command + 1) % SCHEDULER_COMMAND_COUNT;
    auto& command = commands[current_command];

    // Wait for the GPU to finish with all resources for this command.
    Synchronize(current_command);

    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
    };

    // Begin the next command buffer.
    command.render_command_buffer.begin(begin_info);
    command.fence_counter = next_fence_counter++;
    command.use_upload_buffer = false;
}

}  // namespace Vulkan
