/// @file
/// @author uentity
/// @date 24.08.2016
/// @brief kernel signleton declaration
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/error.h>
#include <bs/any_array.h>
#include "logging_subsyst.h"
#include "plugins_subsyst.h"
#include "instance_subsyst.h"
#include "config_subsyst.h"

#include <caf/fwd.hpp>

#include <mutex>

NAMESPACE_BEGIN(blue_sky::kernel)
// forward declare subsystems
NAMESPACE_BEGIN(detail)

struct python_subsyst;
struct radio_subsyst;

NAMESPACE_END(detail)
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
	// kernel generic data storage
	using str_any_map_t = std::map< std::string, str_any_array, std::less<> >;
	str_any_map_t str_key_storage_;

	using idx_any_map_t = std::map< std::string, idx_any_array, std::less<> >;
	idx_any_map_t idx_key_storage_;

	std::mutex sync_storage_;

	// indicator of kernel initialization state
	enum class InitState { NonInitialized, Initialized, Down };
	std::atomic<InitState> init_state_;

	// Python support depends on compile flags and can be 'dumb' or 'real'
	std::unique_ptr<detail::python_subsyst> pysupport_;
	// radio subsystem
	std::unique_ptr<detail::radio_subsyst> radio_ss_;

	kimpl();
	~kimpl();

	auto init_radio() -> error;
	auto shutdown_radio() -> void;

	using type_tuple = tfactory::type_tuple;
	auto find_type(const std::string& key) const -> type_tuple;

	auto str_key_storage(const std::string& key) -> str_any_array&;

	auto idx_key_storage(const std::string& key) -> idx_any_array&;
};

/// Kernel internal singleton
using give_kimpl = singleton<kimpl>;
#define KIMPL ::blue_sky::kernel::give_kimpl::Instance()

NAMESPACE_END(blue_sky::kernel)
