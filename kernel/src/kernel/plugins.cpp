/// @file
/// @author uentity
/// @date 14.01.2019
/// @brief Kernel plugins API impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/kernel/plugins.h>
#include <bs/detail/lib_descriptor.h>
#include "kimpl.h"

NAMESPACE_BEGIN(blue_sky::kernel::plugins)

auto register_plugin(const plugin_descriptor* pd) -> bool {
	return KIMPL.register_plugin(pd, blue_sky::detail::lib_descriptor()).second;
}

auto load_plugin(const std::string& fname, bool init_py_subsyst) -> int {
	return KIMPL.load_plugin(fname, init_py_subsyst);
}

auto load_plugins(void* py_root_module) -> int {
	return KIMPL.load_plugins(py_root_module);
}

auto unload_plugin(const plugin_descriptor& pd) -> void {
	KIMPL.unload_plugin(pd);
}

auto unload_plugins() -> void {
	KIMPL.unload_plugins();
}

auto loaded_plugins() -> plugins_enum {
	plugins_enum res;
	for(const auto& plug_ptr : KIMPL.loaded_plugins_)
		res.emplace_back(plug_ptr.first);
	//res.emplace_back(&KIMPL.kernel_pd_);
	//res.emplace_back(&KIMPL.runtime_pd_);
	return res;
}

auto registered_types() -> types_enum {
	types_enum res;
	for(const auto& elem : KIMPL.types_) {
		res.emplace_back(elem);
	}
	return res;
}

auto plugin_types(const plugin_descriptor& pd) -> types_enum {
	return plugin_types(pd.name);
}

auto plugin_types(const std::string& plugin_name) -> types_enum {
	using plug_name_key = detail::plugins_subsyst::plug_name_key;

	types_enum res;
	auto plt = KIMPL.types_.get< plug_name_key >().equal_range(plugin_name);
	std::copy(plt.first, plt.second, std::back_inserter(res));
	return res;
}

NAMESPACE_END(blue_sky::kernel::plugins)
