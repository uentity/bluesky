/// @file
/// @author uentity
/// @date 24.07.2019
/// @brief BS kernel interprocess connectivity
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "../common.h"
#include "../error.h"
#include <caf/fwd.hpp>

NAMESPACE_BEGIN(blue_sky::kernel::radio)

/// access actor system
BS_API auto system() -> caf::actor_system&;

/// starts or stops transmitter
BS_API auto toggle(bool on = true) -> void;

BS_API auto start_server() -> void;

BS_API auto start_client(const std::string& host) -> error;

BS_API auto publish_link(tree::sp_link L) -> error;
BS_API auto unpublish_link(tree::sp_link L) -> error;

NAMESPACE_END(blue_sky::kernel::radio)
