/// @file
/// @author uentity
/// @date 23.08.2016
/// @brief BlueSky kernel implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/kernel.h>
#include "kernel_impl.h"

namespace blue_sky {

kernel::kernel() : pimpl_(new kernel_impl) {}

kernel::~kernel() {}

void kernel::init() {}

void kernel::cleanup() {}

spdlog::logger& kernel::get_log(const char* name) {
	return kernel_impl::get_log(name);
}

int kernel::load_plugin(const std::string& fname, bool init_py_subsyst) {
	return pimpl_->load_plugin(fname, init_py_subsyst);
}

int kernel::load_plugins(void* py_root_module) {
	return pimpl_->load_plugins(py_root_module);
}

void kernel::unload_plugin(const plugin_descriptor& pd) {
	pimpl_->unload_plugin(pd);
}

void kernel::unload_plugins() {
	pimpl_->unload_plugins();
}

const type_descriptor& kernel::demand_type(const type_descriptor& obj_type) {
	return pimpl_->demand_type(type_tuple(obj_type)).td();
}

bool kernel::register_type(const type_descriptor& td, const plugin_descriptor* pd) {
	return pimpl_->register_type(td, pd);
}

kernel::types_enum kernel::registered_types() const {
	types_enum res;
	for(const auto& elem : pimpl_->obj_fab_) {
		res.emplace_back(elem);
	}
	return res;
}

kernel::types_enum kernel::plugin_types(const plugin_descriptor& pd) const {
	types_enum res;
	auto plt = pimpl_->plugin_types_.equal_range(pd);
	for(auto pos = plt.first; pos != plt.second; ++pos)
		res.emplace_back(*pos);

	return res;
}

kernel::types_enum kernel::plugin_types(const std::string& plugin_name) const {
	const plugin_descriptor key{plugin_name};
	const auto plug_ptr = pimpl_->loaded_plugins_.find(&key);
	if(plug_ptr != pimpl_->loaded_plugins_.end())
		return plugin_types(*plug_ptr->first);
	return {};
}

kernel::plugins_enum kernel::loaded_plugins() const {
	plugins_enum res;
	for(const auto& plug_ptr : pimpl_->loaded_plugins_)
		res.emplace_back(plug_ptr.first);
	res.emplace_back(&pimpl_->kernel_pd_);
	res.emplace_back(&pimpl_->runtime_pd_);
	return res;
}

int kernel::register_instance(const sp_obj& obj) {
	return pimpl_->register_instance(obj);
}

int kernel::free_instance(const sp_obj& obj) {
	return pimpl_->free_instance(obj);
}

kernel::instances_enum kernel::instances(const BS_TYPE_INFO& ti) const {
	return pimpl_->instances(ti);
}

} /* namespace blue_sky */

