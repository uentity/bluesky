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
	return *pimpl_->demand_type(obj_type).td_;
}

bool kernel::register_type(const type_descriptor& td, const plugin_descriptor* pd) {
	return pimpl_->register_type(td, pd);
}

} /* namespace blue_sky */

