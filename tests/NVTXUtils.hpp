#pragma once

#include <nvtx3/nvtx3.hpp>

/**
 * @brief Utilities for NVTX instrumentation.
 */
namespace WP17Scheduler::NVTXUtils {
  /**
   * @brief Generates cycling colors based on an index/event id.
   * @tparam T Index type
   * @param i index value
   * @return nxtv3::rgb Color, one value per index/event.
   */
  template <typename T>
  inline nvtx3::rgb nvtxcolor(T i) {
    constexpr nvtx3::rgb CERNBlue{0, 51, 160};
    uint8_t r = ((i * 23)+ CERNBlue.red)   & 0xFF;
    uint8_t g = ((i * 47)+ CERNBlue.green) & 0xFF;
    uint8_t b = ((i * 71)+ CERNBlue.blue)  & 0xFF;
    return nvtx3::rgb{r, g, b};
  }
}
