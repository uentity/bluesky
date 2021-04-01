/// @file
/// @author uentity
/// @date 22.09.2017
/// @brief BS tree Python bindings
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/python/tree.h>
#include <bs/python/list.h>
#include <bs/python/enum.h>
#include <bs/tree/tree.h>
#include <bs/tree/context.h>

#include <bs/serialize/tree.h>
#include <cereal/archives/json.hpp>
#include <sstream>

#include <pybind11/functional.h>
#include <pybind11/chrono.h>

PYBIND11_MAKE_OPAQUE(blue_sky::tree::context::item_tag);

NAMESPACE_BEGIN(blue_sky::python)
using namespace tree;

using links_v = tree::links_v;
using links_l = std::list<link>;
using nodes_l = std::list<node>;

// [IMPORTANT] point that we must return elements BY VALUE from links vector & list
NAMESPACE_BEGIN(detail)

template<> struct vector_needs_copy<links_v> : std::true_type {};
template<> struct vector_needs_copy<links_l> : std::true_type {};

NAMESPACE_END(detail)

NAMESPACE_BEGIN()

// `walk()` proxy functor to allow nodes & leafs lists modification in Python callback 'pyf'
// accepts pointers to containers instead of references
template<typename R, typename FR, typename Node>
auto py_walk(
	R root, std::function<void (FR, std::list<Node>*, links_v*)> pyf, TreeOpts opts
) {
	walk(std::move(root), [pyf = std::move(pyf)] (FR cur_root, std::list<Node>& nodes, links_v& leafs) {
			// convert references to pointers
			pyf(std::move(cur_root), &nodes, &leafs);
		}, opts
	);
}

NAMESPACE_END()

/*-----------------------------------------------------------------------------
 *  tree Python bindings
 *-----------------------------------------------------------------------------*/
// forward declare binding part from other TU
void py_bind_link(py::module&);
void py_bind_node(py::module&);

void py_bind_tree(py::module& m) {
	///////////////////////////////////////////////////////////////////////////////
	//  common part
	//
	// bind non-arithmetic enums
	py::enum_<Req>(m, "Req")
		.value("Data", Req::Data)
		.value("DataNode", Req::DataNode)
	;

	py::enum_<ReqStatus>(m, "ReqStatus")
		.value("Void", ReqStatus::Void)
		.value("Busy", ReqStatus::Busy)
		.value("OK", ReqStatus::OK)
		.value("Error", ReqStatus::Error)
	;

	py::enum_<Key>(m, "Key")
		.value("ID", Key::ID)
		.value("OID", Key::OID)
		.value("Name", Key::Name)
		.value("Type", Key::Type)
		.value("AnyOrder", Key::AnyOrder)
	;

	// bind arithmetic enums (with binary ops, etc)
	bind_enum_with_ops<Flags>(m, "Flags")
		.value("Plain", Flags::Plain)
		.value("Persistent", Flags::Persistent)
		.value("Disabled", Flags::Disabled)
		.value("LazyLoad", Flags::LazyLoad)
		.export_values()
	;

	bind_enum_with_ops<Event>(m, "Event")
		.value("None", Event::None)
		.value("LinkRenamed", Event::LinkRenamed)
		.value("LinkStatusChanged", Event::LinkStatusChanged)
		.value("LinkInserted", Event::LinkInserted)
		.value("LinkErased", Event::LinkErased)
		.value("LinkDeleted", Event::LinkDeleted)
		.value("DataModified", Event::DataModified)
		.value("DataNodeModified", Event::DataNodeModified)
		.value("All", Event::All)
	;

	bind_enum_with_ops<InsertPolicy>(m, "InsertPolicy")
		.value("AllowDupNames", InsertPolicy::AllowDupNames)
		.value("DenyDupNames", InsertPolicy::DenyDupNames)
		.value("RenameDup", InsertPolicy::RenameDup)
		.value("Merge", InsertPolicy::Merge)
	;

	bind_enum_with_ops<TreeOpts>(m, "TreeOpts")
		.value("Normal"         , TreeOpts::Normal)
		.value("WalkUp"         , TreeOpts::WalkUp)
		.value("Deep"           , TreeOpts::Deep)
		.value("Lazy"           , TreeOpts::Lazy)
		.value("FollowSymLinks" , TreeOpts::FollowSymLinks)
		.value("FollowLazyLinks", TreeOpts::FollowLazyLinks)
		.value("MuteOutputNode" , TreeOpts::MuteOutputNode)
		.value("HighPriority"   , TreeOpts::HighPriority)
		.value("DetachedWorkers", TreeOpts::DetachedWorkers)
		.value("TrackWorkers"   , TreeOpts::TrackWorkers)
	;

	// bind lists of links & nodes as opaque types
	bind_rich_vector<links_v>(m, "links_vector", py::module_local(false));
	bind_list<nodes_l>(m, "nodes_list", py::module_local(false));
	bind_list<links_l>(m, "links_list", py::module_local(false));

	// event
	py::class_<event>(m, "event")
		.def_readonly("params", &event::params)
		.def_readonly("code", &event::code)
		.def("origin_link", &event::origin_link, "If event source is link, return it")
		.def("origin_node", &event::origin_node, "If event source is node, return it")
	;

	///////////////////////////////////////////////////////////////////////////////
	//  link & node
	//
	py_bind_link(m);
	py_bind_node(m);

	///////////////////////////////////////////////////////////////////////////////
	//  tree API
	//
	m.def("abspath", &abspath,
		"lnk"_a, "path_unit"_a = Key::ID, "Get link's absolute path", nogil);
	m.def("find_root", py::overload_cast<link>(&find_root),
		"L"_a, "Return root node of a tree that given link belongs to", nogil);
	m.def("find_root", py::overload_cast<node>(&find_root),
		"N"_a, "Return root node of a tree that given node belongs to", nogil);
	m.def("find_root_handle", py::overload_cast<link>(&find_root_handle),
		"L"_a, "Return handle (link) of a tree root that given link belongs to", nogil);
	m.def("find_root_handle", py::overload_cast<node>(&find_root_handle),
		"L"_a, "Return handle (link) of a tree root that given node belongs to", nogil);
	m.def("convert_path", &convert_path,
		"src_path"_a, "start"_a, "src_path_unit"_a = Key::ID, "dst_path_unit"_a = Key::Name,
		"follow_lazy_links"_a = false,
		"Convert path string from one representation to another (for ex. link IDs -> link names)", nogil
	);
	m.def("deref_path",py::overload_cast<const std::string&, link, Key, TreeOpts>(&deref_path),
		"path"_a, "start"_a, "path_unit"_a = Key::ID, "opts"_a = def_deref_opts,
		"Quick link search by given path relative to `start`", nogil
	);
	m.def("deref_path",py::overload_cast<const std::string&, node, Key, TreeOpts>(&deref_path),
		"path"_a, "start"_a, "path_unit"_a = Key::ID, "opts"_a = def_deref_opts,
		"Quick link search by given path relative to `start`", nogil
	);
	// async deref_path
	m.def("deref_path",
		py::overload_cast<deref_process_f, std::string, link, Key, TreeOpts>(&deref_path),
		"deref_cb"_a, "path"_a, "start"_a, "path_unit"_a = Key::ID, "opts"_a = def_deref_opts,
		"Async quick link search by given path relative to `start`", nogil
	);

	// walk
	// [HINT] pass vectors to be modified as pointers - in this case pybind11 applies reference policy
	using py_walk_links_cb = std::function<void(const link&, links_l*, links_v*)>;
	using py_walk_nodes_cb = std::function<void(const node&, nodes_l*, links_v*)>;

	m.def("walk",
		[](
			link root, py_walk_links_cb cb, TreeOpts opts
		) {
			py_walk(root, std::move(cb), opts);
		},
		"root"_a, "step_f"_a, "opts"_a = def_walk_opts,
		"Walk the tree similar to Python `os.walk()`", nogil
	);
	m.def("walk",
		[](
			node root, py_walk_nodes_cb cb, TreeOpts opts
		) {
			py_walk(std::move(root), std::move(cb), opts);
		},
		"root_node"_a, "step_f"_a, "opts"_a = def_walk_opts,
		"Walk the tree similar to Python `os.walk()` (alternative)", nogil
	);

	// make root link
	m.def("make_root_link", py::overload_cast<std::string_view, std::string, sp_obj>(&make_root_link),
		"link_type"_a = "hard_link", "name"_a = "/", "root_obj"_a = nullptr,
		"If object contains node it's handle will point to returned link", nogil
	);
	m.def("make_root_link", py::overload_cast<sp_obj, std::string, std::string_view>(&make_root_link),
		"root_obj"_a, "name"_a = "/", "link_type"_a = "hard_link",
		"If object contains node it's handle will point to returned link", nogil
	);
	// save/load tree
	py::enum_<TreeArchive>(m, "TreeArchive")
		.value("Text", TreeArchive::Text)
		.value("Binary", TreeArchive::Binary)
		.value("FS", TreeArchive::FS)
	;
	m.def("save_tree", py::overload_cast<link, std::string, TreeArchive, timespan>(&save_tree),
		"root"_a, "filename"_a, "ar"_a = TreeArchive::FS, "wait_for"_a = infinite, nogil);
	m.def("save_tree", py::overload_cast<on_serialized_f, link, std::string, TreeArchive>(&save_tree),
		"callback"_a, "root"_a, "filename"_a, "ar"_a = TreeArchive::FS, nogil);
	m.def("load_tree", py::overload_cast<std::string, TreeArchive>(&load_tree),
		"filename"_a, "ar"_a = TreeArchive::FS, nogil);
	m.def("load_tree", py::overload_cast<on_serialized_f, std::string, TreeArchive>(&load_tree),
		"callback"_a, "filename"_a, "ar"_a = TreeArchive::FS, nogil);

	// dump tree to/from string
	m.def("to_string", [](link r) -> result_or_err<std::string> {
		auto os = std::ostringstream{};
		if(auto er = error::eval_safe([&] {
			cereal::JSONOutputArchive ja(os);
			ja(r);
		}))
			return tl::make_unexpected(er);
		return os.str();
	}, "root"_a, nogil);

	m.def("from_string", [](const std::string& input) -> link_or_err {
		link res;
		if(auto er = error::eval_safe([&] {
			auto os = std::istringstream{input};
			cereal::JSONInputArchive ja(os);
			ja(res);
		}))
			return tl::make_unexpected(er);
		return res;
	}, "root"_a, nogil);

	///////////////////////////////////////////////////////////////////////////////
	//  Qt model helper
	//
	using item_tag = context::item_tag;
	using existing_tag = context::existing_tag;

	auto py_qth = py::class_<context>(m, "tree_context")
		.def(py::init<node>(), "root"_a = node::nil())
		.def(py::init<sp_obj>(), "root_obj"_a)
		.def(py::init<link>(), "root"_a)

		.def_property_readonly("root", &context::root)
		.def_property_readonly("root_link", &context::root_link)
		.def("root_path", &context::root_path, "path_unit"_a = Key::Name)

		.def("reset", py::overload_cast<link>(&context::reset), "root"_a, "Reset context to new root")
		.def("reset", py::overload_cast<node, link>(&context::reset),
			"root"_a, "root_handle"_a = link{}, "Reset context to new root")

		// [NOTE] it's essential to keep returned tags in Python as long as context exists
		// use combined return value policy:
		// py::keep_alive<1, 0>() - keeps returned value while context exists
		// py::return_value_policy::reference - don't destruct C++ tag instance on cleanup
		.def("__call__", py::overload_cast<const std::string&, bool>(&context::operator()),
			"path"_a, "nonexact_match"_a = false,
			py::keep_alive<1, 0>(), py::return_value_policy::reference
		)
		.def("__call__", py::overload_cast<const link&, std::string>(&context::operator()),
			"lnk"_a, "path_hint"_a = "/",
			py::keep_alive<1, 0>(), py::return_value_policy::reference
		)
		.def("__call__", py::overload_cast<std::int64_t, existing_tag>(&context::operator()),
			"row"_a, "parent"_a = existing_tag{},
			py::keep_alive<1, 0>(), py::return_value_policy::reference
		)
		.def("__call__", py::overload_cast<existing_tag>(&context::operator()),
			"child"_a,
			py::keep_alive<1, 0>(), py::return_value_policy::reference
		)

		.def("dump", &context::dump)
	;

	py::class_<item_tag>(py_qth, "item_tag")
		.def_property_readonly("path", [](const item_tag& t) { return to_string(t.first); })
		.def_property_readonly("link", [](const item_tag& t) { return t.second.lock(); })
	;
}

NAMESPACE_END(blue_sky::python)
