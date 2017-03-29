/// @file
/// @author uentity
/// @date 21.10.2016
/// @brief Kernel types factory functions
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/exception.h>
#include <bs/log.h>
#include "kernel_plugins_subsyst.h"

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(detail)

using namespace std;

// static nil elements for td_ptr and fab_elem
//template< > const type_descriptor td_ptr::nil_el = type_descriptor();
//template< > const fab_elem fe_ptr::nil_el = fab_elem();

//bool kernel_plugins_subsyst::register_kernel_type(const type_descriptor& td, fab_elem* tp_ref) {
//	return register_type(td, &kernel_pd_, tp_ref);
//}
//
//bool kernel_plugins_subsyst::register_rt_type(const type_descriptor& td, fab_elem* tp_ref) {
//	return register_type(td, &runtime_pd_, tp_ref);
//}

bool kernel_plugins_subsyst::register_type(
	const type_descriptor& td, const plugin_descriptor* pd, fab_elem* tp_ref
) {
	if(td.is_nil()) return false;

	pair< factory_t::iterator, bool > res;

	// find correct plugin_descriptor
	// pointer to registered plugin_descriptor
	pd_ptr pdp(pd);
	// if now plugin descriptor provided - register as runtime type
	if(pdp.is_nil())
		pdp = runtime_pd_;
	if(!is_inner_pd(*pdp)) {
		// case for external plugin descriptor
		// try to register it or find a match with existing pd
		pdp = register_plugin(pdp.get(), lib_descriptor()).first;
	}
	// register obj in factory
	res = obj_fab_.insert(fab_elem(pdp, td));
	// save registered type if asked for
	if(tp_ref) *tp_ref = *res.first;

	//register type in dictionaries
	if(res.second) {
		pair< types_dict_t::const_iterator, bool > res_ref = types_resolver_.insert(*res.first);
		if(!res_ref.second) {
			//probably duplicating type name found
			// dump error
			bserr() << "kernel:factory: type '" << td.type_name()
				<< "' cannot be registered because type with such name already exist" << bs_end;

			obj_fab_.erase(res.first);
			if(tp_ref) {
				if(res_ref.first != types_resolver_.end())
					*tp_ref = *res_ref.first;
				else
					//some unknown bad error happened
					*tp_ref = fe_ptr::nil_el();
			}
			return false;
		}

		plugin_types_.insert(*res.first);
		return true;
	}
	else if(*pdp != *res.first->pd_ && is_inner_pd(*res.first->pd_)) {
		// current type was previously registered as inner
		// replace with correct plugin d-tor now
		// remove first from types_resolver_
		types_resolver_.erase(*res.first);
		// remove inner-type association
		plugin_types_.erase(*res.first);
		// now delete factory entry
		obj_fab_.erase(res.first);

		// register type with passed plugin_descriptor
		res = obj_fab_.insert(fab_elem(pd, td));
		types_resolver_.insert(*res.first);
		plugin_types_.insert(*res.first);
		return true;
	}

	// dump warning
	BSERROR << log::W("kernel:factory: type '{}' already registered") << td.type_name() << bs_end;
	return false;
}

fab_elem kernel_plugins_subsyst::demand_type(const fab_elem& obj_t) {
	fab_elem tt_ref(obj_t);
	if(obj_t.td_.is_nil()) {
		// if type is nil try to find it by name
		types_dict_t::const_iterator tt = types_resolver_.find(tt_ref);
		if(tt != types_resolver_.end())
			tt_ref = *tt;
	}
	else {
		// otherwise try to find requested type using fast search in factory
		factory_t::const_iterator tt = obj_fab_.find(tt_ref);
		if(tt != obj_fab_.end())
			tt_ref = *tt;
		else
			tt_ref = fe_ptr::nil_el();
	}
	if(tt_ref.td_.is_nil()) {
		//type wasn't found - try to register it first
		if(obj_t.pd_.is_nil())
			register_rt_type(*obj_t.td_, &tt_ref);
		else
			register_type(*obj_t.td_, obj_t.pd_.get(), &tt_ref);
		//still nil td means that serious error happened - type cannot be registered
		if(tt_ref.is_nil()) {
			throw bs_kexception(boost::format(
				"Type (%s) is nil or cannot be registered!") % obj_t.td_->type_name(),
				"factory"
			);
		}
	}
	return tt_ref;
}

NAMESPACE_END(detail)
NAMESPACE_END(blue_sky)

