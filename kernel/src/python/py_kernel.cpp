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
#include <bs/python/any_array.h>
#include <bs/python/map.h>

#include <pybind11/chrono.h>
#include <fmt/format.h>

PYBIND11_MAKE_OPAQUE(blue_sky::str_any_array)
PYBIND11_MAKE_OPAQUE(blue_sky::idx_any_array)
PYBIND11_MAKE_OPAQUE(caf::settings)

NAMESPACE_BEGIN(pybind11::detail)

// bind `caf::variant` as variant-like type
template<typename... Ts>
struct type_caster<caf::variant<Ts...>> : variant_caster<caf::variant<Ts...>> {};

// transparent binder for `caf::config_value` that basically forwards to `caf::variant`
template<>
struct type_caster<caf::config_value> {
	using Type = caf::config_value;
	using variant_type = typename Type::variant_type;

	PYBIND11_TYPE_CASTER(Type, _("config_value"));

	auto load(handle src, bool convert) -> bool {
		auto variant_caster = make_caster<variant_type>();
		if(variant_caster.load(src, convert)) {
			visit_helper<caf::variant>::call(
				[&](auto&& v) { value = std::forward<decltype(v)>(v); },
				cast_op<variant_type>(std::move(variant_caster))
			);
			return true;
		}
		return false;
	}

	template<typename T>
	static auto cast(T&& src, return_value_policy pol, handle parent) {
		return make_caster<variant_type>::cast( std::forward<T>(src).get_data(), pol, parent );
	}
};

NAMESPACE_END(pybind11::detail)

NAMESPACE_BEGIN(blue_sky::python)
using type_tuple = kernel::tfactory::type_tuple;

NAMESPACE_BEGIN()
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
}

auto bind_config_api(py::module& m) -> void {
	m.def("configure", &kernel::config::configure,
		"Configure kernel with CLI arguments and/or config file(s)",
		"cli_args"_a = std::vector<std::string>(), "ini_fname"_a = "", "force"_a = false
	);
	m.def("is_configured", &kernel::config::is_configured);
	m.def("settings", &kernel::config::config, py::return_value_policy::reference);
}

auto bind_misc_api(py::module& m) {
	m.def("last_error", &kernel::last_error);
	m.def("str_key_storage", &kernel::str_key_storage, py::return_value_policy::reference);
	m.def("idx_key_storage", &kernel::idx_key_storage, py::return_value_policy::reference);

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

	bind_rich_map<caf::settings>(m, "settings", py::module_local(false));

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
 *  main kernel binding fns
 *-----------------------------------------------------------------------------*/
void py_bind_kernel(py::module& m) {
	auto kmod = m.def_submodule("kernel", "BS kernel");
	bind_kernel_api(kmod);
}

NAMESPACE_END(blue_sky::python)
