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
#include <bs/serialize/tree.h>
#include <bs/tree/map_link.h>

#include "kernel_queue.h"
#include "../kernel/python_subsyst_impl.h"

#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/operators.h>

NAMESPACE_BEGIN(blue_sky::python)
using namespace tree;

NAMESPACE_BEGIN()

const auto py_kernel = &kernel::detail::python_subsyst_impl::self;

auto adapt(sp_obj&& source, const link& L) {
	return py_kernel().adapt(std::move(source), L);
}

auto adapt(obj_or_err&& source, const link& L) {
	return std::move(source).map([&](sp_obj&& obj) {
		return py_kernel().adapt(std::move(obj), L);
	});
}

using adapted_data_cb = std::function<void(result_or_err<py::object>, link)>;
// pipe callback through kernel's queue
auto adapt(adapted_data_cb&& f) {
	return [f = std::move(f)](result_or_err<sp_obj> obj, link L) mutable {
		KRADIO.enqueue(
			launch_async,
			transaction{[f = std::move(f), obj = std::move(obj), L]() mutable {
				// capture GIL to call `adapt()`
				auto guard = py::gil_scoped_acquire();
				f(adapt(std::move(obj), L), L);
				return perfect;
			}}
		);
	};
}

template<typename F, typename R, typename... Args>
auto mapper2queue(F mf, identity<R (Args...)> _) {
	return [mf = std::move(mf)](Args... args, caf::event_based_actor* worker) -> caf::result<R> {
		auto rp = worker->make_response_promise();
		KRADIO.enqueue(
			launch_async,
			[=, argstup = std::make_tuple(std::forward<Args>(args)...)]() mutable -> error {
				if constexpr(std::is_same_v<R, void>) {
					std::apply(mf, std::move(argstup));
					rp.deliver( caf::unit );
				}
				else
					rp.deliver( std::apply(mf, std::move(argstup)) );
				return perfect;
			}
		);
		return rp;
	};
};


// there's no way in Python to select function overload based an arg types,
// so need to introduce special enum
enum class MappingLevel { Link, Node };

template<MappingLevel L>
auto py_mapper2queue(py::function py_mf) {
	using signature = std::conditional_t<
		L == MappingLevel::Link, map_link::simple_link_mapper_f, map_link::simple_node_mapper_f
	>;
	return mapper2queue(py::cast<signature>(std::move(py_mf)), identity< deduce_callable_t<signature> >{});
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
		.def("data_raw", py::overload_cast<>(&bare_link::data), "Returns non-adapted object")
		.def("data_node", &bare_link::data_node, "Get pointee DataNode")
	;

	add_common_api(bare_link_pyface);

	///////////////////////////////////////////////////////////////////////////////
	//  Base link
	//
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

		// [NOTE] return adapted objects
		.def("data_ex",
			[](const link& L, bool wait_if_busy) {
				// release GIL while obtaining data
				const auto call_data_ex = [&] {
					auto _ = py::gil_scoped_release();
					return L.data_ex(wait_if_busy);
				};
				return adapt(call_data_ex(), L);
			},
			"wait_if_busy"_a = true
		)

		.def("data", [](const link& L) {
			// release GIL while obtaining data
			const auto call_data = [&] {
				auto _ = py::gil_scoped_release();
				return L.data();
			};
			return adapt(call_data(), L);
		})
		.def("data_raw", py::overload_cast<>(&link::data, py::const_), nogil, "Returns non-adapted object")

		.def("data",
			[](const link& L, adapted_data_cb f, bool high_priority) {
				return L.data(adapt(std::move(f)), high_priority);
			},
			"f"_a, "high_priority"_a = false
		)

		.def("data_node_ex", &link::data_node_ex, "wait_if_busy"_a = true, nogil)
		.def("data_node", py::overload_cast<>(&link::data_node, py::const_), nogil)

		.def("data_node",
			[](const link& self, link::process_dnode_cb f, bool hp) {
				self.data_node(pipe_through_queue(std::move(f), launch_async), hp);
			},
			"f"_a, "high_priority"_a = false
		)

		// unsafe access to data & data node
		.def("data", py::overload_cast<unsafe_t>(&link::data, py::const_),
			"Direct access to link's data cache", nogil)
		.def("data_node", py::overload_cast<unsafe_t>(&link::data_node, py::const_),
			"Direct access to link's data node cache", nogil)

		// [NOTE] export only async overload, because otherwise Python will hang when moving
		// callback into actor
		.def("apply",
			[](const link& L, py_link_transaction tr) {
				L.apply(launch_async, pytr_through_queue(std::move(tr)));
			},
			"tr"_a, "Send transaction `tr` to link's queue, return immediately"
		)

		.def("data_apply",
			[](const link& L, py_obj_transaction tr) {
				L.data_apply(launch_async, pytr_through_queue(std::move(tr)));
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

		.def("unsubscribe", py::overload_cast<deep_t>(&link::unsubscribe, py::const_))
		.def("unsubscribe", py::overload_cast<>(&engine::unsubscribe, py::const_))
		// [NOTE] need to define non-static method for overloading
		.def("unsubscribe", [](const py::object&, std::uint64_t handler_id) {
			engine::unsubscribe(handler_id);
		}, "handler_id"_a)
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
	py::enum_<MappingLevel>(m, "MappingLevel")
		.value("Link", MappingLevel::Link)
		.value("Node", MappingLevel::Node)
	;

	static const auto make_map_link = [](MappingLevel mlevel, py::function py_mf, auto&&... args) {
		return map_link(
			[&]() -> map_link::mapper_f {
				if(mlevel == MappingLevel::Link)
					return py_mapper2queue<MappingLevel::Link>(std::move(py_mf));
				else
					return py_mapper2queue<MappingLevel::Node>(std::move(py_mf));
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
					std::move(name), std::move(src), std::move(dst), update_on,
					opts | TreeOpts::TrackWorkers, flags
				);
			}),
			"mlevel"_a, "mf"_a, "name"_a, "src_node"_a, "dest_node"_a = link_or_node{},
			"update_on"_a = Event::DataModified,
			"opts"_a = TreeOpts::Normal | TreeOpts::MuteOutputNode,
			"flags"_a = Flags::Plain
		)
		// normal ctor with tag
		.def(py::init([](
				MappingLevel mlevel, py::function py_mf, uuid tag, std::string name,
				link_or_node src, link_or_node dst, Event update_on, TreeOpts opts, Flags flags
			) {
				return make_map_link(
					mlevel, std::move(py_mf),
					tag, std::move(name), std::move(src), std::move(dst), update_on,
					opts | TreeOpts::TrackWorkers, flags
				);
			}),
			"mlevel"_a, "mf"_a, "tag"_a, "name"_a, "src_node"_a, "dest_node"_a = link_or_node{},
			"update_on"_a = Event::DataModified,
			"opts"_a = TreeOpts::Normal | TreeOpts::MuteOutputNode,
			"flags"_a = Flags::Plain
		)
		// from mapper & existing map_link
		.def(py::init([](
				MappingLevel mlevel, py::function py_mf, const map_link& rhs,
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
		"opts"_a = TreeOpts::Deep | TreeOpts::MuteOutputNode, "flags"_a = Flags::Plain
	);

	///////////////////////////////////////////////////////////////////////////////
	//  link cast
	//
	#define LINK_CAST_IMPL(link_t) if(tgt_type == link_t::type_id_()) return link_cast<link_t>(rhs);

	m.def("link_cast", [](const link& rhs, std::string_view tgt_type) -> std::optional<link> {
		LINK_CAST_IMPL(hard_link)
		LINK_CAST_IMPL(weak_link)
		LINK_CAST_IMPL(sym_link)
		LINK_CAST_IMPL(fusion_link)
		LINK_CAST_IMPL(map_link)
		return {};
	}, "rhs"_a, "tgt_type"_a);
}

NAMESPACE_END(blue_sky::python)
