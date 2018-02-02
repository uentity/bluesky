/// @file
/// @author uentity
/// @date 23.08.2016
/// @brief BlueSky kernel implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/kernel.h>
#include <bs/misc.h>
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
	return pimpl_->demand_type({obj_type}).td();
}

bool kernel::register_type(const type_descriptor& td, const plugin_descriptor* pd) {
	return pimpl_->register_type(td, pd);
}

bool kernel::register_type(const type_descriptor& td, const std::string& plug_name) {
	return pimpl_->register_type(td, plug_name);
}

kernel::types_enum kernel::registered_types() const {
	types_enum res;
	for(const auto& elem : pimpl_->types_) {
		res.emplace_back(elem);
	}
	return res;
}

kernel::types_enum kernel::plugin_types(const plugin_descriptor& pd) const {
	return plugin_types(pd.name);
}

kernel::types_enum kernel::plugin_types(const std::string& plugin_name) const {
	using plug_name_key = detail::kernel_plugins_subsyst::plug_name_key;

	types_enum res;
	auto plt = pimpl_->types_.get< plug_name_key >().equal_range(plugin_name);
	std::copy(plt.first, plt.second, std::back_inserter(res));
	return res;
}

kernel::plugins_enum kernel::loaded_plugins() const {
	plugins_enum res;
	for(const auto& plug_ptr : pimpl_->loaded_plugins_)
		res.emplace_back(plug_ptr.first);
	//res.emplace_back(&pimpl_->kernel_pd_);
	//res.emplace_back(&pimpl_->runtime_pd_);
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

type_tuple kernel::find_type(const std::string& type_name) const {
	return pimpl_->find_type(type_name);
}

str_any_array& kernel::pert_str_any_array(const type_descriptor& master) {
	return pimpl_->pert_str_any_array(master);
}

idx_any_array& kernel::pert_idx_any_array(const type_descriptor& master) {
	return pimpl_->pert_idx_any_array(master);
}

std::string kernel::last_error() const {
	return last_system_message();
}

type_descriptor::shared_ptr_cast kernel::clone_object(bs_type_copy_param source) const {
	return source->bs_resolve_type().clone(source);
}

} /* namespace blue_sky */

