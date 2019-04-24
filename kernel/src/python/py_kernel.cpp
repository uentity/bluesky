/// @file
/// @author uentity
/// @date 08.09.2017
/// @brief Python bindings for BS kernel
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/bs.h>
#include <bs/tree/node.h>
#include <bs/python/kernel.h>
#include <bs/python/any.h>

#include <iterator>

#include <pybind11/chrono.h>
#include <fmt/format.h>

PYBIND11_MAKE_OPAQUE(blue_sky::str_any_array)
PYBIND11_MAKE_OPAQUE(blue_sky::idx_any_array)

NAMESPACE_BEGIN(blue_sky::python)

using type_tuple = kernel::tfactory::type_tuple;

NAMESPACE_BEGIN()

type_tuple test_type_tuple(const type_tuple& t) {
	std::cout << t.type_name() << ' ' << t.plug_name() << std::endl;
	return t;
}

template<template<class> class array_traits, typename T>
void set_any_array_value(
	any_array<array_traits>& A, const typename any_array<array_traits>::key_type& k, T value
) {
	if(A.has_key(k))
		A.template ss<T>(k) = std::move(value);
	else
		A.insert_element(k, std::move(value));
}

template< template<class> class array_traits >
void bind_any_array(py::module& m, const char* type_name) {
	using any_array_t = any_array<array_traits>;
	using key_type = typename any_array_t::key_type;
	using container_t = typename any_array_t::container_t;

	static auto get_size = [](const any_array_t& A){ return A.size(); };

	static auto make_array_key_iterator = [](any_array_t& A) {
		if constexpr(any_array_t::is_map)
			return py::make_key_iterator(A.begin(), A.end());
		else
			return py::make_iterator(A.begin(), A.end());
	};

	auto any_bind = py::class_<any_array_t>(m, type_name)
		.def(py::init<>())
		.def("__bool__",
			[](const any_array_t& A) -> bool { return !A.empty(); },
			"Check whether the array is nonempty"
		)
		.def("__len__", get_size)
		.def_property_readonly("size", get_size)

		.def("__contains__", [](const any_array_t& A, py::handle k) {
			auto key_caster = py::detail::make_caster<key_type>();
			return key_caster.load(k, true) ?
				A.has_key(py::detail::cast_op<key_type>(key_caster)) :
				false;
		})

		.def("__getitem__", [](any_array_t& A, const key_type& k) {
			using array_trait = typename any_array_t::trait;
			auto pval = array_trait::find(A, k);
			if(pval == A.end())
				throw py::key_error("There's no element with key = " + fmt::format("{}", k));
			return array_trait::iter2val(pval);
		}, py::return_value_policy::reference_internal)
		.def("__getitem__", [](any_array_t& A, std::size_t k) {
			if(k >= A.size())
				throw py::key_error("Index past array size");
			using array_trait = typename any_array_t::trait;
			return array_trait::iter2val(std::next(std::begin(A), k));
		}, py::return_value_policy::reference_internal)

		.def("__setitem__", [](any_array_t& A, const key_type& k, std::any value) {
			if(!any_array_t::trait::insert(A, k, std::move(value)))
				throw py::key_error("Cannot insert element with key = " + fmt::format("{}", k));
		})

		.def("__delitem__", [](any_array_t& m, const key_type& k) {
			auto it = any_array_t::trait::find(m, k);
			if (it == m.end()) throw py::key_error();
			m.erase(it);
		})

		.def("__iter__", make_array_key_iterator,
			py::keep_alive<0, 1>() /* Essential: keep list alive while iterator exists */
		)
		.def("items",
			[](any_array_t &m) { return py::make_iterator(m.begin(), m.end()); },
			py::keep_alive<0, 1>() /* Essential: keep list alive while iterator exists */
		)

		.def("keys", [](const any_array_t& A){ return A.keys(); })
		.def("values", [](const any_array_t& A) {
			std::vector<std::any> res;
			for(auto a = std::begin(A), end = std::end(A); a != end; ++a)
				res.push_back(any_array_t::trait::iter2val(a));
			return res;
		})
	;
}

template <typename Map, typename holder_type = std::unique_ptr<Map>, typename... Args>
auto bind_cafdict(py::handle scope, const std::string &name, Args&&... args) {
	using KeyType = typename Map::key_type;
	using MappedType = typename Map::mapped_type;
	using Class_ = py::class_<Map, holder_type>;
	namespace detail = py::detail;
	using namespace py;

	// If either type is a non-module-local bound type then make the map binding non-local as well;
	// otherwise (e.g. both types are either module-local or converting) the map will be
	// module-local.
	auto tinfo = detail::get_type_info(typeid(MappedType));
	bool local = !tinfo || tinfo->module_local;
	if (local) {
		tinfo = detail::get_type_info(typeid(KeyType));
		local = !tinfo || tinfo->module_local;
	}

	Class_ cl(scope, name.c_str(), pybind11::module_local(local), std::forward<Args>(args)...);

	cl.def(init<>());

	// Register stream insertion operator (if possible)
	detail::map_if_insertion_operator<Map, Class_>(cl, name);

	cl.def("__bool__",
		[](const Map &m) -> bool { return !m.empty(); },
		"Check whether the dict is nonempty"
	);

	cl.def("__iter__",
		   [](Map &m) { return make_key_iterator(m.begin(), m.end()); },
		   keep_alive<0, 1>() /* Essential: keep list alive while iterator exists */
	);

	cl.def("items",
		   [](Map &m) { return make_iterator(m.begin(), m.end()); },
		   keep_alive<0, 1>() /* Essential: keep list alive while iterator exists */
	);

	cl.def("__getitem__",
		[](Map &m, const KeyType &k) -> MappedType & {
			auto it = m.find(k);
			if (it == m.end())
			  throw key_error();
		   return it->second;
		},
		return_value_policy::reference_internal // ref + keepalive
	);

	// Assignment provided only if the type is copyable
	detail::map_assignment<Map, Class_>(cl);

	cl.def("__len__", &Map::size);

	return cl;
}

///////////////////////////////////////////////////////////////////////////////
//  bind kernel subsystems API
//
auto bind_plugins_api(py::module& m) -> void {
	m.def("load_plugin", &kernel::plugins::load_plugin, "fname"_a);
	m.def("load_plugins", &kernel::plugins::load_plugins);
	m.def("unload_plugin", &kernel::plugins::unload_plugin,
		"plugin_descriptor"_a);
	m.def("unload_plugins", &kernel::plugins::unload_plugins);
	m.def("loaded_plugins", &kernel::plugins::loaded_plugins);
	m.def("plugin_types", py::overload_cast<const plugin_descriptor&>(&kernel::plugins::plugin_types),
		"plugin_descriptor"_a);
	m.def("plugin_types", py::overload_cast<const std::string&>(&kernel::plugins::plugin_types),
		"plugin_name"_a);
	m.def("registered_types", &kernel::plugins::registered_types);
}

auto bind_tfactory_api(py::module& m) -> void {
	m.def("register_type",
		py::overload_cast<const type_descriptor&, const plugin_descriptor*>(&kernel::tfactory::register_type),
		"type_descriptor"_a, "plugin_descriptor"_a = nullptr);
	m.def("register_type",
		py::overload_cast<const type_descriptor&, const std::string&>(&kernel::tfactory::register_type),
		"type_descriptor"_a, "plugin_name"_a);
	m.def("find_type", &kernel::tfactory::find_type, "type_name"_a);
	m.def("register_instance", (int (*)(sp_cobj)) &kernel::tfactory::register_instance);
	m.def("free_instance", (int (*)(sp_cobj)) &kernel::tfactory::free_instance);
	m.def("instances",
		(auto (*)(const type_descriptor&) -> kernel::tfactory::instances_enum) &kernel::tfactory::instances);
	m.def("instances",
		(auto (*)(std::string_view) -> kernel::tfactory::instances_enum) &kernel::tfactory::instances);
	m.def("clone_object", [](bs_type_copy_param src) -> sp_obj {
		return kernel::tfactory::clone_object(src);
	}, "src"_a, "Make object copy");

	// [TODO] remove
	m.def("test_type_tuple", &test_type_tuple);
}

auto bind_config_api(py::module& m) -> void {
	m.def("configure", &kernel::config::configure,
		"Configure kernel with CLI arguments and/or config file(s)",
		"cli_args"_a = std::vector<std::string>(), "ini_fname"_a = "", "force"_a = false
	);
	m.def("is_configured", &kernel::config::is_configured);
	m.def("config", &kernel::config::config, py::return_value_policy::reference);
}

auto bind_misc_api(py::module& m) {
	m.def("last_error", &kernel::last_error);
	m.def("pert_str_any_array", &kernel::pert_str_any_array, py::return_value_policy::reference);
	m.def("pert_idx_any_array", &kernel::pert_idx_any_array, py::return_value_policy::reference);

	m.def("init", &kernel::init, "Call this manually in the very beginning of main()");
	m.def("shutdown", &kernel::shutdown, "Call this manually before program ends (before exit from main)");
	m.def("unify_serialization", &kernel::unify_serialization);
	m.def("k_descriptor", &kernel::k_descriptor, py::return_value_policy::reference);
}

auto bind_tools(py::module& m) -> void {
	m.def("print_loaded_types", &kernel::tools::print_loaded_types);
	m.def("print_link", [](const tree::sp_link& l) { kernel::tools::print_link(l, 0); });
	m.def("print_link", [](const tree::sp_node& n, std::string name = "/") {
		kernel::tools::print_link(std::make_shared<tree::hard_link>(name, n), 0);
	}, "node"_a, "root_name"_a = "/");
}

auto bind_kernel_api(py::module& m) -> void {
	py::enum_<kernel::Error>(m, "Error")
		.value("OK", kernel::Error::OK)
		.value("CantLoadDLL", kernel::Error::CantLoadDLL)
		.value("CantUnloadDLL", kernel::Error::CantUnloadDLL)
		.value("CantRegisterType", kernel::Error::CantRegisterType)
		.value("TypeIsNil", kernel::Error::TypeIsNil)
		.value("TypeAlreadyRegistered", kernel::Error::TypeAlreadyRegistered)
		.value("CantCreateLogger", kernel::Error::CantCreateLogger)
	;
	BSPY_REGISTER_ERROR_ENUM(kernel::Error)

	bind_any_array<str_any_traits>(m, "str_any_array");
	bind_any_array<idx_any_traits>(m, "idx_any_array");

	py::class_<caf::config_value>(m, "caf_config_value");
	bind_cafdict<caf::config_value::dictionary>(m, "caf_config_dict");
	bind_cafdict<caf::config_value_map>(m, "caf_config_map");

	bind_misc_api(m);

	auto k_subm = m.def_submodule("plugins", "Plugins API");
	bind_plugins_api(k_subm);

	k_subm = m.def_submodule("tfactory", "Types factory API");
	bind_tfactory_api(k_subm);

	k_subm = m.def_submodule("config", "Config API");
	bind_config_api(k_subm);

	k_subm = m.def_submodule("tools", "Kernel tools API");
	bind_tools(k_subm);
}

NAMESPACE_END() // eof hodden namespace

/*-----------------------------------------------------------------------------
 *  main kernel binding fcn
 *-----------------------------------------------------------------------------*/
void py_bind_kernel(py::module& m) {
	auto kmod = m.def_submodule("kernel", "BS kernel");
	bind_kernel_api(kmod);
}

NAMESPACE_END(blue_sky::python)
