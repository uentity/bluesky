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

#include "../kernel/radio_subsyst.h"

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
	m.def("load_plugin", &kernel::plugins::load_plugin, "fname"_a, nogil);
	m.def("load_plugins", &kernel::plugins::load_plugins, nogil);
	m.def("unload_plugin", &kernel::plugins::unload_plugin, "plugin_descriptor"_a, nogil);
	m.def("unload_plugins", &kernel::plugins::unload_plugins, nogil);
	m.def("loaded_plugins", &kernel::plugins::loaded_plugins, nogil);
	m.def("plugin_types", py::overload_cast<const plugin_descriptor&>(&kernel::plugins::plugin_types),
		"plugin_descriptor"_a, nogil);
	m.def("plugin_types", py::overload_cast<const std::string&>(&kernel::plugins::plugin_types),
		"plugin_name"_a, nogil);
	m.def("registered_types", &kernel::plugins::registered_types, nogil);
}

auto bind_tfactory_api(py::module& m) -> void {
	m.def("register_type",
		py::overload_cast<const type_descriptor&, const plugin_descriptor*>(&kernel::tfactory::register_type),
		"type_descriptor"_a, "plugin_descriptor"_a = nullptr);
	m.def("register_type",
		py::overload_cast<const type_descriptor&, const std::string&>(&kernel::tfactory::register_type),
		"type_descriptor"_a, "plugin_name"_a);
	m.def("find_type", &kernel::tfactory::find_type, "type_name"_a);

	m.def("create", [](std::string_view obj_type_id) -> sp_obj {
		return kernel::tfactory::create(obj_type_id);
	}, "obj_type_id"_a, "Default construct object of given type");

	m.def("clone", [](bs_type_copy_param src) -> sp_obj {
		return kernel::tfactory::clone(src);
	}, "src"_a, "Make object copy");

	m.def("assign", &kernel::tfactory::assign,
		"target"_a, "source"_a, "params"_a = prop::propdict{}, "Assign content from source to target");
	// assign that works on links level
	m.def("assign", [](tree::link tar, tree::link src, prop::propdict params) -> error {
		if(tar && src) {
			return tar.data_apply([&](sp_obj tar_obj) -> tr_result {
				return kernel::tfactory::assign(tar_obj, src.data());
			});
		}
		return { "assign: either source or target link is nil" };
	}, "target"_a, "source"_a, "params"_a = prop::propdict{}, "Assign pointee from source to target link");
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
	m.def("shutdown", &kernel::shutdown, "Call this manually before program ends (before exit from main)", nogil);
	m.def("unify_serialization", &kernel::unify_serialization);
	m.def("k_descriptor", &kernel::k_descriptor, py::return_value_policy::reference);
}

auto bind_tools(py::module& m) -> void {
	m.def("print_loaded_types", &kernel::tools::print_loaded_types);
	m.def("print_link", [](const tree::link& l) { kernel::tools::print_link(l, 0); }, nogil);
	m.def("print_link", [](tree::node n, std::string name = "/") {
		kernel::tools::print_link(tree::hard_link(name, std::move(n)), 0);
	}, "node"_a, "root_name"_a = "/", nogil);
}

auto bind_radio(py::module& m) -> void {
	namespace kr = kernel::radio;
	m.def("await_all_actors_done", [] { kr::system().await_all_actors_done(); },
		"Blocks until all actors are finished", nogil);
	m.def("await_actors_before_shutdown", [] {return kr::system().await_actors_before_shutdown(); });
	m.def("await_actors_before_shutdown", [](bool x) { kr::system().await_actors_before_shutdown(x); });
	m.def("toggle", &kr::toggle, "Start/stop radio subsystem");
	m.def("start_server", &kr::start_server, "Helper that toggles radio on & starts listening", nogil);
	m.def("start_client", &kr::start_client, "Helper that toggles radio on & starts client", nogil);
	m.def("publish_link", &kr::publish_link, nogil);
	m.def("unpublish_link", &kr::unpublish_link, nogil);
	m.def("bye_actor", &kr::bye_actor, "actor_id"_a, "Send `a_bye` message to registered actor with given ID");
	m.def("kick_citizens", [] { KRADIO.kick_citizens(); });
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

	k_subm = m.def_submodule("radio", "Distribution API");
	bind_radio(k_subm);
}

NAMESPACE_END() // eof hodden namespace

/*-----------------------------------------------------------------------------
 *  main kernel binding fns
 *-----------------------------------------------------------------------------*/
auto py_bind_adapters(py::module& m) -> void;

auto py_bind_kernel(py::module& m) -> void {
	// place adapters into root bs module
	py_bind_adapters(m);

	auto kmod = m.def_submodule("kernel", "BS kernel");
	bind_kernel_api(kmod);
}

NAMESPACE_END(blue_sky::python)
