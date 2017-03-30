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

//#define BS_AUTOLOAD_PLUGINS

/*!
	\brief Macro for registering your objects in blue-sky.

	Generates proper call to kernel's register_type function.

	\param plugin_descr - descriptor of your plugin. Can be taken, for example, from plugin_initializer that is passed to bs_register_plugin
	\param T = class name
	\param creation_fun = creation function pointer
 */
#define BLUE_SKY_REGISTER_TYPE(plugin_descr, T) \
blue_sky::give_kernel::Instance().register_type(plugin_descr, T::bs_type())

//! \defgroup kernel_group kernel group - kernel of blue-sky
namespace blue_sky {

namespace detail {
	struct wrapper_kernel;
}

// type tuple - contains type information coupled with plugin information
struct type_tuple : public std::tuple< const plugin_descriptor*, const type_descriptor* > {
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
	// no way to register rvalue (temp) type_descriptor
	bool register_type(type_descriptor&&, const plugin_descriptor* = nullptr) = delete;

	// create instance of object
	template< typename Obj_type_spec, typename... Args >
	auto create_object(Obj_type_spec&& obj_type, Args&&... ctor_args) {
		const type_descriptor& td = demand_type(type_descriptor(std::forward< Obj_type_spec >(obj_type)));
		return td.construct(std::forward< Args >(ctor_args)...);
	}

	// clone object
	auto create_object_copy(bs_type_copy_param source) const {
		return source->bs_resolve_type().clone(source);
	}

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
		return register_instance(std::static_pointer_cast< objbase >(obj));
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
		return free_instance(std::static_pointer_cast< objbase >(obj));
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

} // eof blue_sky namespace

