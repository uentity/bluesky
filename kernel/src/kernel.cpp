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

#include <spdlog/spdlog.h>

namespace blue_sky {

kernel::kernel() : pimpl_(new kernel_impl) {}

kernel::~kernel() {}

void kernel::init() {
	using InitState = kernel_impl::InitState;

	// do initialization only once from non-initialized state
	auto expected_state = InitState::NonInitialized;
	if(pimpl_->init_state_.compare_exchange_strong(expected_state, InitState::Initialized)) {
		// configure kernel
		pimpl_->configure();
		// switch to mt logs
		pimpl_->toggle_mt_logs(true);
		// init actor system
		auto& actor_sys = pimpl_->actor_sys_;
		if(!actor_sys) {
			actor_sys = std::make_unique<caf::actor_system>(pimpl_->actor_cfg_);
			if(!actor_sys)
				throw error("Can't create CAF actor_system!");
		}
	}
}

void kernel::shutdown() {
	using InitState = kernel_impl::InitState;

	// shut down if not already Down
	if(pimpl_->init_state_.exchange(InitState::Down) != InitState::Down) {
		// destroy actor system
		if(pimpl_->actor_sys_) {
			pimpl_->actor_sys_.release();
		}
		// shutdown mt logs
		pimpl_->toggle_mt_logs(false);
		spdlog::shutdown();
	}
}

auto kernel::configure(
	std::vector<std::string> args, std::string ini_fname, bool force
) -> const caf::config_value_map& {
	pimpl_->configure(std::move(args), ini_fname, force);
	return pimpl_->confdata_;
}

auto kernel::config() const -> const caf::config_value_map& {
	return pimpl_->confdata_;
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

int kernel::register_instance(sp_cobj obj) {
	return pimpl_->register_instance(std::move(obj));
}

int kernel::free_instance(sp_cobj obj) {
	return pimpl_->free_instance(std::move(obj));
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

const plugin_descriptor& kernel::self_descriptor() const {
	return pimpl_->kernel_pd();
}

void* kernel::self_pymod() const {
	return pimpl_->self_pymod();
}

bool kernel::register_plugin(const plugin_descriptor* pd) {
	return pimpl_->register_plugin(pd, detail::lib_descriptor()).second;
}

void kernel::unify_serialization() const {
	pimpl_->unify_serialization();
}

caf::actor_system_config& kernel::actor_config() const {
	return pimpl_->actor_cfg_;
}

caf::actor_system& kernel::actor_system() const {
	return pimpl_->actor_system();
}

} /* namespace blue_sky */

