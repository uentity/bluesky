/// @file
/// @author uentity
/// @date 24.08.2016
/// @brief Plugins subsystem of BS kernel
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include <bs/kernel.h>
#include"lib_descriptor.h"
#include <boost/pool/pool_alloc.hpp>
#include <map>

namespace blue_sky { namespace detail {
// hidden details
namespace {
/*-----------------------------------------------------------------------------
 *  helper wrapper for storing objects in kernel dictionary
 *  designed to have specific valid nill element
 *-----------------------------------------------------------------------------*/
template< class T >
struct elem_ptr {
	typedef T elem_t;
	//nil ctor
	elem_ptr() : p_(&nil_el) {}

	elem_ptr(const T& el) : p_(&el) {}

	elem_ptr (const T *el) : p_ (el) {}

	elem_ptr& operator =(const T& el) {
		p_ = &el;
		return *this;
	}

	const T* operator ->() const { return p_; }

	//conversion to elem_t
	const T& elem() const { return *p_; }
	const T& operator *() const { return *p_; }

	bool is_nil() const {
		return p_->is_nil();
	}

	operator bool() const {
		return !is_nil();
	}

	static const T nil_el;

private:
	const T* p_;
};
// alias
using pd_ptr = elem_ptr< plugin_descriptor >;
template< > const plugin_descriptor pd_ptr::nil_el = plugin_descriptor();

/*-----------------------------------------------------------------------------
 *  main type factory element
 *  contains pair of plugin_descriptor and type_descriptor wrapped by elem_ptr
 *-----------------------------------------------------------------------------*/
//typedef type_tuple fab_elem;
struct fab_elem {
	// nil ctor
	fab_elem() {}

	fab_elem(const pd_ptr& pd, const type_descriptor& td)
		: pd_(pd), td_(td)
	{}
	//the same but from tuple
	fab_elem(const type_tuple& tt)
		: pd_(tt.pd), td_(tt.td)
	{}

	fab_elem(const pd_ptr& pd)
		: pd_(pd)
	{}

	fab_elem(const type_descriptor& td)
		: td_(td)
	{}

	bool is_nil() const {
		return (td_.is_nil() && pd_.is_nil());
	}

	operator bool() const {
		return !is_nil();
	}

	operator type_tuple() const {
		return type_tuple(*pd_, td_);
	};

	pd_ptr pd_;
	type_descriptor td_;
};
// alias
using fe_ptr = elem_ptr< fab_elem >;
// static nil elements for fab_elem
template< > const fab_elem fe_ptr::nil_el = fab_elem();

// pd_ptrs comparison by name
struct pdp_comp_name {
	bool operator()(const pd_ptr& lhs, const pd_ptr& rhs) const {
		return (lhs->name < rhs->name);
	}
};

// fab_elem sorting order by bs_type_info
struct fe_comp_ti {
	bool operator()(const fab_elem& lhs, const fab_elem& rhs) const {
		return (lhs.td_ < rhs.td_);
	}
};

// fab_elem pointers comparison functions for sorting in containers
struct fep_comp_ti {
	bool operator()(const fe_ptr& lhs, const fe_ptr& rhs) const {
		return (lhs->td_ < rhs->td_);
	}
};

// fab element comparison by string typename
struct fep_comp_stype {
	bool operator()(const fe_ptr& lhs, const fe_ptr& rhs) const {
		return (lhs->td_.type_name() < rhs->td_.type_name());
	}
};

// fab elements comparison by plugin name
struct fep_comp_pd {
	bool operator()(const fe_ptr& lhs, const fe_ptr& rhs) const {
		return (lhs->pd_->name < rhs->pd_->name);
	}
};

// compare plugin_descriptors by name
struct cmp_pd_by_name {
	bool operator()(
		const blue_sky::plugin_descriptor& lhs,
		const blue_sky::plugin_descriptor& rhs
	) {
		return (lhs.name < rhs.name);
	}
};

} // eof hidden namespace

/*-----------------------------------------------------------------
 * Kernel plugins loading subsystem definition
 *----------------------------------------------------------------*/
struct BS_HIDDEN_API kernel_plugins_subsyst {

	const plugin_descriptor kernel_pd_;       //! plugin descriptor for kernel types
	const plugin_descriptor runtime_pd_;      //! plugin descriptor for runtime types

	//! plugin_descriptor -> lib_descriptor 1-to-1 relation
	using plugins_enum_t = std::map<
		plugin_descriptor, lib_descriptor, cmp_pd_by_name,
		boost::fast_pool_allocator<
			std::pair< plugin_descriptor, lib_descriptor >,
			boost::default_user_allocator_new_delete,
			boost::details::pool::null_mutex
		>
	>;

	//! plugin_descriptors sorted by name
	using plugins_dict_t = std::multiset<
		pd_ptr, pdp_comp_name,
		boost::fast_pool_allocator<
			pd_ptr, boost::default_user_allocator_new_delete, boost::details::pool::null_mutex
		>
	>;

	//! types factory: fab_elements sorted by BS_TYPE_INFO
	using factory_t = std::set<
		fab_elem, fe_comp_ti,
		boost::fast_pool_allocator<
			fab_elem, boost::default_user_allocator_new_delete, boost::details::pool::null_mutex
		>
	>;

	//! types dictionary: fe_ptrs sorted by string type
	using types_dict_alloc_t = boost::fast_pool_allocator<
		fe_ptr, boost::default_user_allocator_new_delete, boost::details::pool::null_mutex
	>;
	using types_dict_t = std::set<
		fe_ptr, fep_comp_stype, types_dict_alloc_t
	>;

	//! loaded plugins: pointers to type_tuples sorted by plugins names
	using plugin_types_enum_t = std::multiset<
		fe_ptr, fep_comp_pd, types_dict_alloc_t
	>;

	plugins_enum_t loaded_plugins_;
	plugins_dict_t plugins_dict_;
	plugin_types_enum_t plugin_types_;

	factory_t obj_fab_;
	types_dict_t types_resolver_;

	// ctor
	kernel_plugins_subsyst();

	void clean_plugin_types(const plugin_descriptor& pd);

	void unload_plugin(const plugin_descriptor& pd);

	void unload_plugins();

	std::pair< pd_ptr, bool > register_plugin(const plugin_descriptor& pd, const lib_descriptor& ld);

	int load_plugin(const std::string& fname, bool init_py_subsyst);
};

}} // eof hidden namespace

