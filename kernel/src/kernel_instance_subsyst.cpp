/// @file
/// @author uentity
/// @date 15.03.2017
/// @brief Kernel instances subsystem implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "kernel_instance_subsyst.h"

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(detail)

// register instance of any BlueSky type
int kernel_instance_subsyst::register_instance(sp_cobj&& obj) {
	if(!obj) return 0;
	const type_descriptor* td = &obj->bs_resolve_type();
	// go through chain of type_descriptors up to objbase
	int arity = 0;
	while(!td->is_nil()) {
		arity += instances_[td->type()].insert(std::move(obj)).second;
		td = &td->parent_td();
	}
	return arity;
}

int kernel_instance_subsyst::free_instance(sp_cobj&& obj) {
	if(!obj) return 0;

	// go through chain of type_descriptors up to objbase
	const type_descriptor* td = &obj->bs_resolve_type();
	int arity = 0;
	while(!td->is_nil()) {
		arity += (int)instances_[td->type()].erase(std::move(obj));
		td = &td->parent_td();
	}
	return arity;
}

kernel::instances_enum kernel_instance_subsyst::instances(const BS_TYPE_INFO& ti) const {
	auto Is = instances_.find(ti);
	if(Is == instances_.end())
		return {};

	instances_enum res;
	res.reserve(Is->second.size());
	std::copy(Is->second.begin(), Is->second.end(), std::back_inserter(res));
	return res;
}

NAMESPACE_END(detail)
NAMESPACE_END(blue_sky)

