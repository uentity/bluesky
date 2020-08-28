/// @file
/// @author uentity
/// @date 22.04.2019
/// @brief BS tree links bindings
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/python/tree.h>
#include <bs/python/result_converter.h>
#include "../kernel/python_subsyst_impl.h"

#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/operators.h>

NAMESPACE_BEGIN(blue_sky::python)
using namespace tree;

NAMESPACE_BEGIN()

inline const auto py_kernel = &kernel::detail::python_subsyst_impl::self;

auto adapt(sp_obj&& source, const link& L) {
	auto guard = py::gil_scoped_acquire();
	return py_kernel().adapt(std::move(source), L);
}

template<typename T>
auto adapt(result_or_err<T>&& source, const link& L) {
	return std::move(source).map([&](T&& obj) {
		auto guard = py::gil_scoped_acquire();
		if constexpr(std::is_same_v<T, sp_obj>)
			return py_kernel().adapt( std::move(obj), L );
		else
			return py_kernel().adapt( std::static_pointer_cast<objbase>(std::move(obj)), L );
	});
}

using adapted_data_cb = std::function<void(result_or_err<py::object>, link)>;

inline auto adapt(adapted_data_cb&& f) {
	return [f = std::move(f)](result_or_err<sp_obj> obj, link L) {
		if(L)
			f(adapt(std::move(obj), L), L);
		else
			f(tl::make_unexpected(error{ "Bad (null) link" }), L);
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
	//  engine
	//
	py::class_<engine>(m, "engine")
		.def_property_readonly("type_id", &engine::type_id)
		.def_property_readonly("home_id", &engine::home_id)
		.def(py::self == py::self)
		.def(py::self != py::self)
		.def(py::self < py::self)
		.def(hash(py::self))
	;

	///////////////////////////////////////////////////////////////////////////////
	//  Base link
	//
	using py_modificator_f = std::function< py::object(sp_obj) >;

	// link base class
	auto link_pyface = py::class_<link, engine>(m, "link")
		.def(py::init())
		.def(py::init<std::string, sp_obj, Flags>(), "name"_a, "data"_a, "f"_a = Plain)
		.def(py::init<std::string, node, Flags>(), "name"_a, "folder"_a, "f"_a = Plain)

		.def("__bool__", [](const link& self) { return (bool)self; }, py::is_operator())

		.def_property_readonly("is_nil", [](const link& self) { return self.is_nil(); })
		.def_property_readonly("id", &link::id)
		.def_property_readonly("owner", &link::owner)

		.def("clone", &link::clone, "deep"_a = false, "Make shallow or deep copy of link", nogil)

		// [NOTE] return adapted objects (and pass 'em to callbacks)
		.def("data_ex",
			[](const link& L, bool wait_if_busy) {
				return adapt(L.data_ex(wait_if_busy), L);
			},
			"wait_if_busy"_a = true, nogil
		)
		.def("data", [](const link& L){ return adapt(L.data(), L); }, nogil)
		.def("data", [](const link& L, adapted_data_cb f, bool high_priority) {
				return L.data(adapt(std::move(f)), high_priority);
			}, "f"_a, "high_priority"_a = false, nogil
		)
		.def("data", py::overload_cast<unsafe_t>(&link::data, py::const_), "Direct access to link's data cache")

		.def("data_node_ex", &link::data_node_ex, "wait_if_busy"_a = true, nogil)
		.def("data_node", py::overload_cast<>(&link::data_node, py::const_), nogil)
		.def("data_node", py::overload_cast<link::process_dnode_cb, bool>(&link::data_node, py::const_),
			"f"_a, "high_priority"_a = false, nogil
		)
		.def("data_node", py::overload_cast<unsafe_t>(&link::data_node, py::const_), "Direct access to link's data node cache")

		// [NOTE] export only async overload, because otherwise Python will hang when moving
		// callback into actor
		.def("data_apply",
			[](const link& L, py_modificator_f m, bool silent) {
				L.data_apply(launch_async, make_result_converter<error>(std::move(m), perfect), silent);
			},
			"m"_a, "silent"_a = false,
			"Place given modificator `m` to object's queue and return immediately", nogil
		)

		.def("oid", py::overload_cast<>(&link::oid, py::const_), nogil)
		.def("oid", py::overload_cast<unsafe_t>(&link::oid, py::const_))

		.def("obj_type_id", py::overload_cast<>(&link::obj_type_id, py::const_), nogil)
		.def("obj_type_id", py::overload_cast<unsafe_t>(&link::obj_type_id, py::const_))

		.def("rename", py::overload_cast<std::string>(&link::rename, py::const_))
		.def("req_status", &link::req_status, "Query for given operation status", nogil)
		.def("rs_reset", &link::rs_reset,
			"request"_a, "new_status"_a = ReqStatus::Void,
			"Unconditionally set status of given request", nogil
		)
		.def("rs_reset_if_eq", &link::rs_reset_if_eq,
			"request"_a, "self_rs"_a, "new_rs"_a = ReqStatus::Void,
			"Set status of given request if it is equal to given value, returns prev status", nogil
		)
		.def("rs_reset_if_neq", &link::rs_reset_if_neq,
			"request"_a, "self_rs"_a, "new_rs"_a = ReqStatus::Void,
			"Set status of given request if it is NOT equal to given value, returns prev status", nogil
		)

		.def("name", py::overload_cast<>(&link::name, py::const_), nogil)
		.def_property_readonly("name_unsafe", [](const link& L) { return L.name(unsafe); })

		.def("flags", py::overload_cast<>(&link::flags, py::const_), nogil)
		.def_property_readonly("flags_unsafe", [](const link& L) { return L.flags(unsafe); })
		.def("set_flags", &link::set_flags)

		.def("info", py::overload_cast<>(&link::info, py::const_), nogil)
		.def_property_readonly("info_unsafe", [](const link& L) { return L.info(unsafe); })

		.def("is_node", &link::is_node, "Check if pointee is a node", nogil)
		.def("data_node_hid", py::overload_cast<>(&link::data_node_hid, py::const_),
			"If pointee is a node, return node's actor group ID", nogil)
		.def("data_node_hid", py::overload_cast<unsafe_t>(&link::data_node_hid, py::const_))

		// events subscrition
		.def("subscribe", &link::subscribe, "event_cb"_a, "events"_a = Event::All, nogil)
		.def_static("unsubscribe", &link::unsubscribe, "event_cb_id"_a)
	;

	// link::weak_ptr
	bind_weak_ptr(link_pyface);

	///////////////////////////////////////////////////////////////////////////////
	//  Derived links
	//
	py::class_<hard_link, link>(m, "hard_link")
		.def(py::init<std::string, sp_obj, Flags>(),
			"name"_a, "data"_a, "flags"_a = Flags::Plain)
		.def(py::init<const link&>())
		.def_property_readonly_static("type_id_", [](const py::object&) { return hard_link::type_id_(); })
	;

	py::class_<weak_link, link>(m, "weak_link")
		.def(py::init<std::string, const sp_obj&, Flags>(),
			"name"_a, "data"_a, "flags"_a = Flags::Plain)
		.def(py::init<const link&>())
		.def_property_readonly_static("type_id_", [](const py::object&) { return weak_link::type_id_(); })
	;

	py::class_<sym_link, link>(m, "sym_link")
		.def(py::init<std::string, std::string, Flags>(),
			"name"_a, "path"_a, "flags"_a = Flags::Plain)
		.def(py::init<std::string, const link&, Flags>(),
			"name"_a, "source"_a, "flags"_a = Flags::Plain)
		.def(py::init<const link&>())
		.def_property_readonly_static("type_id_", [](const py::object&) { return sym_link::type_id_(); })

		.def_property_readonly("check_alive", &sym_link::check_alive, nogil)
		.def("target", &sym_link::target, "Get target that sym link points to", nogil)
		.def("target_path", &sym_link::target_path, "human_readable"_a = false, nogil)
	;

	///////////////////////////////////////////////////////////////////////////////
	//  fusion link/iface
	//
	py::class_<fusion_link, link>(m, "fusion_link")
		.def(py::init<std::string, sp_obj, sp_fusion, Flags>(),
			"name"_a, "data"_a, "bridge"_a = nullptr, "flags"_a = Flags::Plain)
		.def(py::init<std::string, const char*, std::string, sp_fusion, Flags>(),
			"name"_a, "obj_type"_a, "oid"_a = "", "bridge"_a = nullptr, "flags"_a = Flags::Plain)
		.def(py::init<const link&>())
		.def_property_readonly_static("type_id_", [](const py::object&) { return fusion_link::type_id_(); })

		.def_property("bridge", &fusion_link::bridge, &fusion_link::reset_bridge, nogil)

		.def("populate",
			py::overload_cast<const std::string&, bool>(&fusion_link::populate, py::const_),
			"child_type_id"_a, "wait_if_busy"_a = true, nogil
		)
		.def("populate",
			py::overload_cast<link::process_dnode_cb, std::string>(&fusion_link::populate, py::const_),
			"f"_a, "obj_type_id"_a, nogil
		)
	;

	py::class_<fusion_iface, py_fusion<>, std::shared_ptr<fusion_iface>>(m, "fusion_iface")
		.def("populate", &fusion_iface::populate, "root"_a, "root_link"_a, "child_type_id"_a = "",
			"Populate root object structure (children)")
		.def("pull_data", &fusion_iface::pull_data, "root"_a, "root_link"_a,
			"Fill root object content")
	;
}

NAMESPACE_END(blue_sky::python)
