/// @file
/// @author uentity
/// @date 22.04.2019
/// @brief BS tree links bindings
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/python/common.h>
#include <bs/python/tree.h>
#include <bs/tree/link.h>
#include <bs/tree/node.h>
#include "../kernel/python_subsyst_impl.h"

#include <boost/uuid/uuid_io.hpp>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

NAMESPACE_BEGIN(blue_sky::python)
using namespace tree;

NAMESPACE_BEGIN()

auto py_kernel() -> kernel::detail::python_subsyst_impl& {
	return kernel::detail::python_subsyst_impl::self();
}

auto adapt(sp_obj&& source) {
	return py_kernel().adapt(std::move(source));
}

template<typename T>
auto adapt(result_or_err<T>&& source) {
	return std::move(source).map([](T&& obj) {
		if constexpr(std::is_same_v<T, sp_obj>)
			return py_kernel().adapt( std::move(obj) );
		else
			return py_kernel().adapt( std::static_pointer_cast<objbase>(std::move(obj)) );
	});
}

using adapted_data_cb = std::function<void(result_or_err<py::object>, sp_clink)>;

auto adapt(adapted_data_cb&& f) {
	return [f = std::move(f)](result_or_err<sp_obj> obj, sp_clink L) {
		f(adapt(std::move(obj)), std::move(L));
	};
}

NAMESPACE_END()

void py_bind_link(py::module& m) {
	///////////////////////////////////////////////////////////////////////////////
	//  inode
	//
	py::class_<inode>(m, "inode")
		.def_readonly("owner", &inode::owner)
		.def_readonly("group", &inode::group)
		.def_readonly("mod_time", &inode::mod_time)
		.def_property_readonly("flags", [](const inode& i) { return i.flags; })
		.def_property_readonly("u", [](const inode& i) { return i.u; })
		.def_property_readonly("g", [](const inode& i) { return i.g; })
		.def_property_readonly("o", [](const inode& i) { return i.o; })
	;

	///////////////////////////////////////////////////////////////////////////////
	//  Base link
	//
	py::class_<link, py_link<>, std::shared_ptr<link>> link_pyface(m, "link");

	// export Flags enum
	py::enum_<link::Flags>(link_pyface, "Flags", py::arithmetic())
		.value("Persistent", link::Flags::Persistent)
		.value("Disabled", link::Flags::Disabled)
		.export_values()
	;
	py::implicitly_convertible<int, link::Flags>();
	py::implicitly_convertible<long, link::Flags>();

	// export link request & status enums
	py::enum_<link::Req>(link_pyface, "Req")
		.value("Data", link::Req::Data)
		.value("DataNode", link::Req::DataNode)
	;
	py::enum_<link::ReqStatus>(link_pyface, "ReqStatus")
		.value("Void", link::ReqStatus::Void)
		.value("Busy", link::ReqStatus::Busy)
		.value("OK", link::ReqStatus::OK)
		.value("Error", link::ReqStatus::Error)
	;

	// Events enum
	py::enum_<Event>(m, "Event")
		.value("LinkRenamed", Event::LinkRenamed)
		.value("LinkStatusChanged", Event::LinkStatusChanged)
		.value("LinkInserted", Event::LinkInserted)
		.value("LinkErased", Event::LinkErased)
		.value("All", Event::All)
	;

	// link base class
	link_pyface
		.def("clone", &link::clone, "deep"_a = false, "Make shallow or deep copy of link")

		// [NOTE] return adapted objects (and pass 'em to callbacks)
		.def("data_ex",
			[](const link& L, bool wait_if_busy) { return adapt(L.data_ex(wait_if_busy)); },
			"wait_if_busy"_a = true
		)
		.def("data", [](const link& L){ return adapt(L.data()); })
		.def("data", [](const link& L, adapted_data_cb f, bool high_priority) {
				return L.data(adapt(std::move(f)), high_priority);
			}, "f"_a, "high_priority"_a = false
		)

		.def("data_node_ex", &link::data_node_ex, "wait_if_busy"_a = true)
		.def("data_node", py::overload_cast<>(&link::data_node, py::const_))
		.def("data_node", py::overload_cast<link::process_data_cb, bool>(&link::data_node, py::const_),
			"f"_a, "high_priority"_a = false
		)

		.def("type_id", &link::type_id)
		.def("oid", &link::oid)
		.def("obj_type_id", &link::obj_type_id)
		.def("rename", &link::rename)
		.def("req_status", &link::req_status, "Query for given operation status")
		.def("rs_reset", &link::rs_reset,
			"request"_a, "new_status"_a = link::ReqStatus::Void,
			"Unconditionally set status of given request"
		)
		.def("rs_reset_if_eq", &link::rs_reset_if_eq,
			"request"_a, "self_rs"_a, "new_rs"_a = link::ReqStatus::Void,
			"Set status of given request if it is equal to given value, returns prev status"
		)
		.def("rs_reset_if_neq", &link::rs_reset_if_neq,
			"request"_a, "self_rs"_a, "new_rs"_a = link::ReqStatus::Void,
			"Set status of given request if it is NOT equal to given value, returns prev status"
		)
		.def("rs_data_changed", &link::rs_data_changed, "Signal tree that pointee has been midified")

		.def_property_readonly("id", [](const link& L) {
			return boost::uuids::to_string(L.id());
		})
		.def_property_readonly("name", &link::name)
		.def_property_readonly("owner", &link::owner)
		.def_property_readonly("info", &link::info)
		.def_property("flags", &link::flags, &link::set_flags)

		// events subscrition
		.def("subscribe", &link::subscribe, "event_cb"_a, "events"_a = Event::All)
		.def("unsubscribe", &link::unsubscribe, "event_cb_id"_a)
	;

	// export adapters manip functions
	using adapter_fn = kernel::detail::python_subsyst_impl::adapter_fn;

	m.def("register_adapter", [](std::string obj_type_id, adapter_fn f) {
			py_kernel().register_adapter(std::move(obj_type_id), std::move(f));
		}, "obj_type_id"_a, "adapter_fn"_a, "Register adapter for specified BS type"
	);
	m.def("register_default_adapter", [](adapter_fn f) {
			py_kernel().register_default_adapter(std::move(f));
		}, "adapter_fn"_a, "Register default adapter for all BS types with no adapter registered"
	);
	m.def("clear_adapters", []() { py_kernel().clear_adapters(); },
		"Remove all adapters (including default) for BS types"
	);
	m.def("adapt", [](sp_obj obj) { return py_kernel().adapt(std::move(obj)); },
		"Make adapter for given object"
	);
	m.def("adapted_types", []() { return py_kernel().adapted_types(); },
		"Return list of types with registered adapters ('*' denotes default adapter)"
	);
	m.def("drop_adapted_cache", [](const std::string& obj_id) {
			return py_kernel().drop_adapted_cache(obj_id);
		}, "obj_id"_a = "",
		"Clear cached adapter for given OID (or drop all cached adapters if OID is empty)"
	);
	m.def("get_cached_adapter", [](const std::string& obj_type_id) {
			return py_kernel().get_cached_adapter(obj_type_id);
		},
		"obj_type_id"_a, "Get cached adapter for object with given OID (if created before, otherwise None)"
	);

	///////////////////////////////////////////////////////////////////////////////
	//  Derived links
	//
	py::class_<hard_link, link, py_link<hard_link>, std::shared_ptr<hard_link>>(m, "hard_link")
		.def(py::init<std::string, sp_obj, link::Flags>(),
			"name"_a, "data"_a, "flags"_a = link::Flags::Plain)
	;

	py::class_<weak_link, link, py_link<weak_link>, std::shared_ptr<weak_link>>(m, "weak_link")
		.def(py::init<std::string, const sp_obj&, link::Flags>(),
			"name"_a, "data"_a, "flags"_a = link::Flags::Plain)
	;

	py::class_<sym_link, link, py_link<sym_link>, std::shared_ptr<sym_link>>(m, "sym_link")
		.def(py::init<std::string, std::string, link::Flags>(),
			"name"_a, "path"_a, "flags"_a = link::Flags::Plain)
		.def(py::init<std::string, const sp_link&, link::Flags>(),
			"name"_a, "source"_a, "flags"_a = link::Flags::Plain)

		.def_property_readonly("check_alive", &sym_link::check_alive)
		.def("src_path", &sym_link::src_path, "human_readable"_a = false)
	;

	///////////////////////////////////////////////////////////////////////////////
	//  fusion link/iface
	//
	py::class_<fusion_link, link, py_link<fusion_link>, std::shared_ptr<fusion_link>>(m, "fusion_link")
		.def(py::init<std::string, sp_node, sp_fusion, link::Flags>(),
			"name"_a, "data"_a, "bridge"_a = nullptr, "flags"_a = link::Flags::Plain)
		.def(py::init<std::string, const char*, std::string, sp_fusion, link::Flags>(),
			"name"_a, "obj_type"_a, "oid"_a = "", "bridge"_a = nullptr, "flags"_a = link::Flags::Plain)
		.def_property("bridge", &fusion_link::bridge, &fusion_link::reset_bridge)
		.def("populate",
			py::overload_cast<const std::string&, bool>(&fusion_link::populate, py::const_),
			"child_type_id"_a, "wait_if_busy"_a = true
		)
		.def("populate",
			py::overload_cast<link::process_data_cb, std::string>(&fusion_link::populate, py::const_),
			"f"_a, "obj_type_id"_a
		)
	;

	py::class_<fusion_iface, py_fusion<>, std::shared_ptr<fusion_iface>>(m, "fusion_iface")
		.def("populate", &fusion_iface::populate, "root"_a, "child_type_id"_a = "",
			"Populate root object structure (children)")
		.def("pull_data", &fusion_iface::pull_data, "root"_a,
			"Fill root object content")
	;
}

NAMESPACE_END(blue_sky::python)
