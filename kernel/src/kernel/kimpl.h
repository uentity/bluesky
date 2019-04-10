/// @file
/// @author uentity
/// @date 24.08.2016
/// @brief kernel signleton declaration
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/any_array.h>
#include "logging_subsyst.h"
#include "plugins_subsyst.h"
#include "instance_subsyst.h"
#include "config_subsyst.h"

#include <caf/fwd.hpp>

NAMESPACE_BEGIN(blue_sky::kernel)
/*-----------------------------------------------------------------------------
 *  kernel impl
 *-----------------------------------------------------------------------------*/
class BS_HIDDEN_API kimpl :
	public detail::config_subsyst,
	public detail::plugins_subsyst,
	public detail::instance_subsyst,
	public detail::logging_subsyst
{
public:
	// per-type kernel any arrays storage
	using str_any_map_t = std::map< std::string, str_any_array, std::less<> >;
	str_any_map_t str_any_map_;

	using idx_any_map_t = std::map< std::string, idx_any_array, std::less<> >;
	idx_any_map_t idx_any_map_;

	// kernel's actor system
	// delayed actor system initialization
	std::unique_ptr<caf::actor_system> actor_sys_;

	// indicator of kernel initialization state
	enum class InitState { NonInitialized, Initialized, Down };
	std::atomic<InitState> init_state_;

	kimpl();
	~kimpl();

	using type_tuple = tfactory::type_tuple;
	auto find_type(const std::string& key) const -> type_tuple;

	auto pert_str_any_array(const std::string& master) -> str_any_array&;

	auto pert_idx_any_array(const std::string& master) -> idx_any_array&;

	auto actor_system() -> caf::actor_system&;
};

/// Kernel internal singleton
using give_kimpl = singleton<kimpl>;
#define KIMPL ::blue_sky::kernel::give_kimpl::Instance()

NAMESPACE_END(blue_sky::kernel)
