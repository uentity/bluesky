/// @file
/// @author uentity
/// @date 14.07.2019
/// @brief Code shared among different BS actors
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/actor_common.h>
#include <bs/atoms.h>

#define OMIT_OBJ_SERIALIZATION                                                          \
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(::blue_sky::error::box)                                   \
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(::blue_sky::sp_obj)                                       \
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(::blue_sky::tree::inodeptr)                               \
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(::blue_sky::tree::link)                                   \
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(::blue_sky::tree::sp_node)                                \
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(::blue_sky::result_or_errbox<::blue_sky::sp_obj>)         \
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(::blue_sky::result_or_errbox<::blue_sky::tree::inodeptr>) \
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(::blue_sky::result_or_errbox<::blue_sky::tree::link>)     \
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(::blue_sky::result_or_errbox<::blue_sky::tree::sp_node>)  \
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(::blue_sky::tree::links_v)                                \
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(::blue_sky::tree::lids_v)

NAMESPACE_BEGIN(blue_sky::tree)

inline constexpr auto def_data_timeout = timespan{ std::chrono::seconds(3) };
inline const std::string nil_grp_id = "<null>";

NAMESPACE_END(blue_sky::tree)
