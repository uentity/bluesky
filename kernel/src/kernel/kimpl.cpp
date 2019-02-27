/// @file
/// @author uentity
/// @date 14.01.2019
/// @brief kernel singleton definition
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/error.h>
#include <bs/kernel/misc.h>
#include "kimpl.h"

#include <caf/actor_system.hpp>
#include <fmt/format.h>

NAMESPACE_BEGIN(blue_sky) NAMESPACE_BEGIN(kernel)

kimpl::kimpl()
	: init_state_(InitState::NonInitialized)
{
	// [NOTE] We can't create `caf::actor_system` here.
	// `actor_system` starts worker and other service threads in constructor.
	// At the same time kernel singleton is constructed most of the time during
	// initialization of kernel shared library. And on Windows it is PROHIBITED to start threads
	// in `DllMain()`, because that cause a deadlock.
	// Solution: delay construction of actor_system until first usage, don't use CAf in kernel
	// ctor.
}

kimpl::~kimpl() = default;

auto kimpl::find_type(const std::string& key) const -> type_tuple {
	using search_key = plugins_subsyst::type_name_key;

	auto tp = types_.get< search_key >().find(key);
	return tp != types_.get< search_key >().end() ? *tp : type_tuple();
}

// returns valid (non-nill) type info
auto kimpl::find_type_info(const type_descriptor& master) const -> BS_TYPE_INFO {
	using type_name_key = plugins_subsyst::type_name_key;

	BS_TYPE_INFO info = master.type();
	if(is_nil(info)) {
		// try to find type info by type name
		info = find_type(master.name).td().type();
	}
	// sanity
	if(is_nil(info))
		throw error(
			fmt::format("Cannot find type info for type {}, seems like not registered", master.name)
		);
	return info;
}

auto kimpl::pert_str_any_array(const type_descriptor& master) -> str_any_array& {
	return str_any_map_[find_type_info(master)];
}

auto kimpl::pert_idx_any_array(const type_descriptor& master) -> idx_any_array& {
	return idx_any_map_[find_type_info(master)];
}

auto kimpl::actor_system() -> caf::actor_system& {
	// delayed actor system initialization
	// [TODO] write safer code
	static auto* actor_sys = [this]{
		init();
		return actor_sys_.get();
	}();
	return *actor_sys;
}

NAMESPACE_END(kernel)

/*-----------------------------------------------------------------------------
 *  kernel internal signleton instantiation
 *-----------------------------------------------------------------------------*/
template<>
BS_API auto singleton<kernel::kimpl>::Instance() -> kernel::kimpl& {
	static kernel::kimpl K;
	return K;
}

NAMESPACE_END(blue_sky)
