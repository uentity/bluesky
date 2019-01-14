/// @file
/// @author uentity
/// @date 14.01.2019
/// @brief Misc kernel API (init/cleanup/... etc)
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "../common.h"
#include "../any_array.h"

NAMESPACE_BEGIN(blue_sky::kernel)

/// kernel initialization routine
/// call this AFTER initialization of statics is done (adter `DllMain()` exit on Windows)
/// this is mostly needed on Windows where omitting this call may lead to deadlock
BS_API auto init() -> void;

/// executable should call this function before program ends (before exit from `main()`)
/// this is mostly needed on Windows where omitting this call will lead to hangup on exit
BS_API auto shutdown() -> void;

/// unite bindings maps for polymorphic serialization code among all loaded plugins
/// such that they can utilize serialization from each other
/// function can be called multiple times, for ex. when new plugin is registered
BS_API auto unify_serialization() -> void;

/// provide access to kernel's plugin_descriptor
BS_API auto k_descriptor() -> const plugin_descriptor&;
/// ... and to kernel's pybind11::module object
BS_API auto k_pymod() -> void*;

BS_API auto last_error() -> std::string;

// access per-type storage that can contain arbitrary types
BS_API auto pert_str_any_array(const type_descriptor& master) -> str_any_array&;
BS_API auto pert_idx_any_array(const type_descriptor& master) -> idx_any_array&;

NAMESPACE_END(blue_sky::kernel)
