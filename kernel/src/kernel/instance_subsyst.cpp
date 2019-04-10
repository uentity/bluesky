/// @file
/// @author uentity
/// @date 15.03.2017
/// @brief Kernel instances subsystem implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/objbase.h>
#include <bs/kernel/types_factory.h>
#include "instance_subsyst.h"

NAMESPACE_BEGIN(blue_sky::kernel::detail)

// register instance of any BlueSky type
auto instance_subsyst::register_instance(sp_cobj&& obj) -> int {
	if(!obj) return 0;
	const type_descriptor* td = &obj->bs_resolve_type();
	// go through chain of type_descriptors up to objbase
	auto play_solo = std::lock_guard(solo_);
	int arity = 0;
	while(!td->is_nil()) {
		arity += instances_[td->name].insert(std::move(obj)).second;
		td = &td->parent_td();
	}
	return arity;
}

auto instance_subsyst::free_instance(sp_cobj&& obj) -> int {
	if(!obj) return 0;
	const type_descriptor* td = &obj->bs_resolve_type();
	// go through chain of type_descriptors up to objbase
	auto play_solo = std::lock_guard(solo_);
	int arity = 0;
	while(!td->is_nil()) {
		arity += (int)instances_[td->name].erase(std::move(obj));
		td = &td->parent_td();
	}
	return arity;
}

auto instance_subsyst::instances(std::string_view type_id) const -> instances_enum {
	auto Is = instances_.find(type_id);
	if(Is == instances_.end())
		return {};

	instances_enum res;
	res.reserve(Is->second.size());
	std::copy(Is->second.begin(), Is->second.end(), std::back_inserter(res));
	return res;
}

NAMESPACE_END(blue_sky::kernel::detail)
