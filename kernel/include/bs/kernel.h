/// @file
/// @author uentity
/// @date 10.08.2016
/// @brief BlueSky kernel class declaration (API)
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include "common.h"
#include "objbase.h"
#include "log.h"
#include "type_descriptor.h"
#include "plugin_descriptor.h"
#include "any_array.h"

NAMESPACE_BEGIN(blue_sky)

namespace detail {
	struct wrapper_kernel;
}

// type tuple - contains type information coupled with plugin information
struct BS_API type_tuple : public std::tuple< const plugin_descriptor*, const type_descriptor* > {
	using base_t = std::tuple< const plugin_descriptor*, const type_descriptor* >;
	using pd_t = const plugin_descriptor&;
	using td_t = const type_descriptor&;

	// construct from lvalue refs to plugin_descriptor & type_descriptor
	template<
		typename P, typename T,
		typename = std::enable_if_t<
			!std::is_rvalue_reference<P&&>::value && !std::is_rvalue_reference<T&&>::value
		>
	>
	type_tuple(P&& plug, T&& type) : base_t(&plug, &type) {}

	// ctors accepting only plugin_descriptor or only type_descriptor
	// uninitialized value will be nil
	type_tuple(pd_t plug) : base_t(&plug, &type_descriptor::nil()) {};
	type_tuple(td_t type) : base_t(&plugin_descriptor::nil(), &type) {}
	// deny constructing from rfavlue refs
	type_tuple(plugin_descriptor&&) = delete;
	type_tuple(type_descriptor&&) = delete;

	// empty ctor creates nil type_tuple
	type_tuple() : base_t(&plugin_descriptor::nil(), &type_descriptor::nil()) {}

	bool is_nil() const {
		return pd().is_nil() && td().is_nil();
	}

	pd_t pd() const {
		return *std::get< 0 >(*this);
	}
	td_t td() const {
		return *std::get< 1 >(*this);
	}

	// direct access to plugin & type names
	// simplifies boost::multi_index_container key specification
	const std::string& plug_name() const {
		return pd().name;
	}
	const std::string& type_name() const {
		return td().name;
	}
};

/*! \class kernel
	*  \ingroup kernel_group
	*  \brief makes access to blue-sky structures
	*
	*         blue-sky kernel makes access to main blue-sky
	*         factory of objects and factory of commands
	*/
class BS_API kernel {
	friend struct detail::wrapper_kernel;

public:
	using plugins_enum = std::vector< const plugin_descriptor* >;
	using types_enum = std::vector< type_tuple >;
	using instances_enum = std::vector< sp_obj >;

	//! \brief Deletes all dangling objects that aren't contained in data storage
	//! This is a garbage collection method
	//! Call it to force system cleaning, although all should be cleaned automatically
	ulong tree_gc() const;

	//! Direct register plugin if shared lib is already loaded
	bool register_plugin(const plugin_descriptor* pd);
	//! \brief Dynamically loads particular plugin
	int load_plugin(const std::string& fname, bool init_py_subsyst);
	//! \brief Load blue-sky plugins method
	int load_plugins(void* py_root_module = nullptr);
	//! \brief Unloads plugin
	void unload_plugin(const plugin_descriptor& pd);
	//! \brief Unload blue-sky plugins method
	void unload_plugins();

	std::string last_error() const;

	static spdlog::logger& get_log(const char* name);

	bool register_type(const type_descriptor& td, const plugin_descriptor* pd = nullptr);
	// register type with plugin_descriptor specified by name
	// types will bind to real plugin descriptor when it will be loaded
	bool register_type(const type_descriptor& td, const std::string& plug_name);
	// no way to register rvalue (temp) type_descriptor
	bool register_type(type_descriptor&&, const plugin_descriptor* = nullptr) = delete;
	bool register_type(type_descriptor&&, const std::string&) = delete;
	// Find type by type name
	type_tuple find_type(const std::string& type_name) const;

	// create instance of object
	template< typename Obj_type_spec, typename... Args >
	auto create_object(Obj_type_spec&& obj_type, Args&&... ctor_args) {
		const type_descriptor& td = demand_type(type_descriptor(std::forward< Obj_type_spec >(obj_type)));
		return td.construct(std::forward< Args >(ctor_args)...);
	}

	// clone object
	type_descriptor::shared_ptr_cast clone_object(bs_type_copy_param source) const;

	// access to plugins & types from them
	//! \brief loaded plugins
	plugins_enum loaded_plugins() const;
	//! \brief registered type infos of objects
	types_enum registered_types() const;
	//! \brief types of plugin (by plugin descriptor)
	types_enum plugin_types(const plugin_descriptor& pd) const;
	types_enum plugin_types(const std::string& plugin_name) const;

	// store and access instances of BS types
	int register_instance(const sp_obj& obj);
	template< class T > int register_instance(const std::shared_ptr< T >& obj) {
		// check that T is inherited from objbase
		static_assert(
			std::is_base_of< objbase, T >::value,
			"Only blue_sky::objbase derived instances can be registered!"
		);
		return register_instance(obj);
		//return register_instance(std::static_pointer_cast< objbase >(obj));
	}
	template< class T > int register_instance(T* obj) {
		// check that T is inherited from objbase
		static_assert(
			std::is_base_of< objbase, T >::value,
			"Only blue_sky::objbase derived instances can be registered!"
		);
		return register_instance(obj->shared_from_this());
	}

	int free_instance(const sp_obj& obj);
	template< class T > int free_instance(const std::shared_ptr< T >& obj) {
		// check that T is inherited from objbase
		static_assert(
			std::is_base_of< objbase, T >::value,
			"Only blue_sky::objbase derived instances can be registered!"
		);
		return free_instance(obj);
		//return free_instance(std::static_pointer_cast< objbase >(obj));
	}
	template< class T > int free_instance(T* obj) {
		// check that T is inherited from objbase
		static_assert(
			std::is_base_of< objbase, T >::value,
			"Only blue_sky::objbase derived instances can be registered!"
		);
		return free_instance(obj->shared_from_this());
	}

	instances_enum instances(const BS_TYPE_INFO& obj_t) const;
	template< typename T > instances_enum instances() const {
		return instances(BS_GET_TI(T));
	}

	// access per-type storage that can contain arbitrary types
	str_any_array& pert_str_any_array(const type_descriptor& master);
	idx_any_array& pert_idx_any_array(const type_descriptor& master);

	/// provide access to kernel's plugin_descriptor
	const plugin_descriptor& self_descriptor() const;

private:
	//! \brief Constructor of kernel
	kernel();
	//! \brief Copy constructor of kernel
	kernel(const kernel&) = delete;
	kernel(kernel&&) = delete;
	//! \brief Destructor.
	~kernel();

	//! PIMPL for kernel
	class kernel_impl;
	std::unique_ptr< kernel_impl > pimpl_;

	// kernel initialization routine
	void init();
	// called before kernel dies
	void cleanup();

	// extract type_descriptor for given type from internal kernel storage
	const type_descriptor& demand_type(const type_descriptor& obj_type);
};

//! \brief singleton for accessing the instance of kernel
typedef singleton< kernel > give_kernel;

NAMESPACE_END(blue_sky)

