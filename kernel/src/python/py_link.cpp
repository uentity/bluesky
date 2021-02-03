/// @file
/// @author uentity
/// @date 22.04.2019
/// @brief BS tree links bindings
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/python/tree.h>
#include <bs/python/tr_result.h>
#include <bs/python/result_converter.h>
#include <bs/detail/enumops.h>

#include <bs/tree/map_link.h>
#include "kernel_queue.h"

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
			f(tl::make_unexpected(error{ "Nil link" }), L);
	};
}

NAMESPACE_END()

void py_bind_link(py::module& m) {
	using namespace allow_enumops;

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
	// minimal link API present in both bare & normal link
	//
	static const auto add_common_api = [](auto& pyl, const auto&... gil) {
		using py_link_type = std::remove_reference_t<decltype(pyl)>;
		using link_type = typename py_link_type::type;

		return pyl
		.def(hash(py::self))
		.def("__bool__", [](const link_type& self) { return (bool)self; }, py::is_operator())

		.def_property_readonly("is_nil", [](const link_type& self) { return self.is_nil(); })
		.def_property_readonly("id", &link_type::id)
		.def_property_readonly("owner", &link_type::owner)

		.def("flags", &link_type::flags, gil...)
		.def("oid", &link_type::oid, gil...)
		.def("obj_type_id", &link_type::obj_type_id, gil...)
		.def("info", &link_type::info, gil...)

		.def("req_status", &link_type::req_status, "Query given operation status")
		.def("data_node_hid", &link_type::data_node_hid, "Get pointee home grop ID")
		;
	};

	///////////////////////////////////////////////////////////////////////////////
	//  Bare link
	//
	auto bare_link_pyface = py::class_<bare_link>(m, "bare_link")
		.def(py::init<const link&>())

		.def_property_readonly("type_id", &bare_link::type_id)
		.def("armed", &bare_link::armed, "Convert to safe link")

		.def("name", &bare_link::name)

		.def("data", [](bare_link& L){ return adapt(L.data(), L.armed()); }, "Get pointee Data")
		.def("data_node", &bare_link::data_node, "Get pointee DataNode")
	;

	add_common_api(bare_link_pyface);

	///////////////////////////////////////////////////////////////////////////////
	//  Base link
	//
	using py_transaction = std::function< py::object() >;
	using py_obj_transaction = std::function< py::object(sp_obj) >;
	using py_link_transaction = std::function< py::object(bare_link) >;

	// link base class
	auto link_pyface = py::class_<link, engine>(m, "link")
		.def(py::init(), "Construct nil link")
		.def(py::init<const bare_link&>())
		.def(py::init<std::string, sp_obj, Flags>(), "name"_a, "data"_a, "f"_a = Plain)
		.def(py::init<std::string, node, Flags>(), "name"_a, "folder"_a, "f"_a = Plain)

		.def("bare", &link::bare, "Convert to bare (unsafe) link")

		.def("clone", &link::clone, "deep"_a = false, "Make shallow or deep copy of link")

		.def("name", py::overload_cast<>(&link::name, py::const_))
		.def_property_readonly("name_unsafe", [](const link& L) { return L.name(unsafe); })

		.def("rename", py::overload_cast<std::string>(&link::rename, py::const_))

		.def("set_flags", &link::set_flags)

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
		// [TODO] figure out how to enable overloads based on transaction arguments
		//.def("apply",
		//	[](const link& L, py_transaction tr) {
		//		L.apply(launch_async, make_result_converter<tr_result>(std::move(tr), {}));
		//	},
		//	"tr"_a, "Send transaction `tr` to link's queue, return immediately"
		//)
		.def("apply",
			[](const link& L, py_link_transaction tr) {
				L.apply(launch_async, make_result_converter<tr_result>(std::move(tr), {}));
			},
			"tr"_a, "Send transaction `tr` to link's queue, return immediately"
		)

		//.def("data_apply",
		//	[](const link& L, py_transaction tr) {
		//		L.data_apply(launch_async, make_result_converter<tr_result>(std::move(tr), {}));
		//	},
		//	"tr"_a, "Send transaction `tr` to object's queue, return immediately"
		//)
		.def("data_apply",
			[](const link& L, py_obj_transaction tr) {
				L.data_apply(launch_async, make_result_converter<tr_result>(std::move(tr), {}));
			},
			"tr"_a, "Send transaction `tr` to object's queue, return immediately"
		)

		.def("data_touch",
			&link::data_touch, "tres"_a = prop::propdict{},
			"Send empty transaction object to trigger `data modified` signal"
		)

		.def("rs_reset", &link::rs_reset,
			"request"_a, "new_status"_a = ReqStatus::Void,
			"Unconditionally set status of given request"
		)
		.def("rs_reset_if_eq", &link::rs_reset_if_eq,
			"request"_a, "self_rs"_a, "new_rs"_a = ReqStatus::Void,
			"Set status of given request if it is equal to given value, returns prev status"
		)
		.def("rs_reset_if_neq", &link::rs_reset_if_neq,
			"request"_a, "self_rs"_a, "new_rs"_a = ReqStatus::Void,
			"Set status of given request if it is NOT equal to given value, returns prev status"
		)

		.def("is_node", &link::is_node, "Check if pointee is a node", nogil)
		.def("data_node_hid", py::overload_cast<>(&link::data_node_hid, py::const_),
			"If pointee is a node, return node's actor group ID", nogil)

		// events subscrition
		.def("subscribe", [](const link& L, link::event_handler f, Event listen_to) {
			return L.subscribe(pipe_through_queue(std::move(f), launch_async), listen_to);
		}, "event_cb"_a, "events"_a = Event::All)

		.def_static("unsubscribe", &link::unsubscribe, "event_cb_id"_a)
	;

	// add mixins
	add_common_api(link_pyface);
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

		.def_property("bridge", &fusion_link::bridge, &fusion_link::reset_bridge)

		.def("pull_data",
			py::overload_cast<prop::propdict, bool>(&fusion_link::pull_data, py::const_),
			"params"_a, "wait_if_busy"_a = true, nogil
		)
		.def("pull_data",
			py::overload_cast<link::process_data_cb, prop::propdict>(&fusion_link::pull_data, py::const_),
			"f"_a, "obj_type_id"_a, nogil
		)

		.def("populate",
			py::overload_cast<prop::propdict, bool>(&fusion_link::populate, py::const_),
			"child_type_id"_a, "wait_if_busy"_a = true, nogil
		)
		.def("populate",
			py::overload_cast<link::process_dnode_cb, prop::propdict>(&fusion_link::populate, py::const_),
			"f"_a, "obj_type_id"_a, nogil
		)
	;

	py::class_<fusion_iface, py_fusion<>, std::shared_ptr<fusion_iface>>(m, "fusion_iface")
		.def("populate", &fusion_iface::populate, "root"_a, "root_link"_a, "params"_a = prop::propdict{},
			"Populate root object structure (children)")
		.def("pull_data", &fusion_iface::pull_data, "root"_a, "root_link"_a, "params"_a = prop::propdict{},
			"Fill root object content")
	;

	///////////////////////////////////////////////////////////////////////////////
	//  map_link
	//
	// there's no way in Python to select function overload based an arg types, so need to introduce
	// additional enum
	enum class MappingLevel { Link, Node };
	py::enum_<MappingLevel>(m, "MappingLevel")
		.value("Link", MappingLevel::Link)
		.value("Node", MappingLevel::Node)
	;

	static const auto make_map_link = [](MappingLevel mlevel, py::function py_mf, auto&&... args) {
		return map_link(
			[&]() -> map_link::mapper_f {
				if(mlevel == MappingLevel::Link)
					return py::cast<map_link::link_mapper_f>(std::move(py_mf));
				else
					return py::cast<map_link::node_mapper_f>(std::move(py_mf));
			}(), std::forward<decltype(args)>(args)...
		);
	};

	py::class_<map_link, link>(m, "map_link")
		// normal ctor
		.def(py::init([](
				MappingLevel mlevel, py::function py_mf, std::string name, link_or_node src, link_or_node dst,
				Event update_on, TreeOpts opts, Flags flags
			) {
				return make_map_link(
					mlevel, std::move(py_mf),
					std::move(name), std::move(src), std::move(dst), update_on, opts, flags
				);
			}),
			"mlevel"_a, "mf"_a, "name"_a, "src_node"_a, "dest_node"_a = link_or_node{},
			"update_on"_a = Event::DataModified, "opts"_a = TreeOpts::DetachedWorkers,
			"flags"_a = Flags::Plain
		)
		// normal ctor with tag
		.def(py::init([](
				MappingLevel mlevel, py::function py_mf, uuid tag, std::string name,
				link_or_node src, link_or_node dst, Event update_on, TreeOpts opts, Flags flags
			) {
				return make_map_link(
					mlevel, std::move(py_mf),
					tag, std::move(name), std::move(src), std::move(dst), update_on, opts, flags
				);
			}),
			"mlevel"_a, "mf"_a, "tag"_a, "name"_a, "src_node"_a, "dest_node"_a = link_or_node{},
			"update_on"_a = Event::DataModified, "opts"_a = TreeOpts::DetachedWorkers,
			"flags"_a = Flags::Plain
		)
		// from mapper & existing map_link
		.def(py::init([](
				MappingLevel mlevel, py::function py_mf, const link& rhs,
				link_or_node src, link_or_node dst
			) {
				return make_map_link(
					mlevel, std::move(py_mf), rhs, std::move(src), std::move(dst)
				);
			}),
			"mlevel"_a, "mf"_a, "rhs"_a, "src_node"_a, "dest_node"_a = link_or_node{}
		)
		// conversion ctor
		.def(py::init<const link&>())

		.def_property_readonly_static("type_id_", [](const py::object&) { return map_link::type_id_(); })

		.def_property_readonly("tag", &map_link::tag)
		.def_property_readonly("input", &map_link::input)
		.def_property_readonly("output", &map_link::output)
	;

	m.def(
		"make_otid_filter", &make_otid_filter, "allowed_otids"_a, "name"_a, "src_node"_a,
		"dest_node"_a = link_or_node{}, "update_on"_a = Event::DataNodeModified | Event::LinkRenamed,
		"opts"_a = TreeOpts::Deep | TreeOpts::DetachedWorkers, "flags"_a = Flags::Plain
	);
}

NAMESPACE_END(blue_sky::python)
