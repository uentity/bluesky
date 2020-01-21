/// @file
/// @author uentity
/// @date 22.09.2017
/// @brief BS tree Python bindings
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/python/common.h>
#include <bs/python/list.h>
#include <bs/tree/tree.h>

#include <pybind11/functional.h>
#include <pybind11/chrono.h>

// make it possible to bind opaque std::list & std::vector (w/o content copying)
PYBIND11_MAKE_OPAQUE(std::vector<blue_sky::tree::sp_link>);
PYBIND11_MAKE_OPAQUE(std::list<blue_sky::tree::sp_link>);
PYBIND11_MAKE_OPAQUE(std::list<blue_sky::tree::sp_node>);

NAMESPACE_BEGIN(blue_sky::python)
using namespace tree;

using links_v = tree::links_v;
using links_l = std::list<sp_link>;
using nodes_l = std::list<sp_node>;

NAMESPACE_BEGIN()

// `walk()` proxy functor to allow nodes & leafs lists modification in Python callback 'pyf'
// accepts pointers to containers instead of references
template<typename R, typename FR, typename Node>
auto py_walk(
	const R& root, std::function<void (const FR&, std::list<Node>*, links_v*)> pyf,
	bool topdown = true, bool follow_symlinks = true, bool follow_lazy_links = false
) {
	walk(root, [pyf = std::move(pyf)] (const FR& cur_root, std::list<Node>& nodes, links_v& leafs) {
			// convert references to pointers
			pyf(cur_root, &nodes, &leafs);
		},
		topdown, follow_symlinks, follow_lazy_links
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
	// invoke link & node bindings
	py_bind_link(m);
	py_bind_node(m);

	///////////////////////////////////////////////////////////////////////////////
	//  tree API
	//
	m.def("abspath", py::overload_cast<const sp_clink&, Key>(&abspath),
		"lnk"_a, "path_unit"_a = Key::ID, "Get link's absolute path");
	m.def("find_root", py::overload_cast<const sp_link&>(&find_root),
		"L"_a, "Return root node of a tree that given link belongs to");
	m.def("find_root", py::overload_cast<sp_node>(&find_root),
		"N"_a, "Return root node of a tree that given node belongs to");
	m.def("find_root_handle", py::overload_cast<sp_clink>(&find_root_handle),
		"L"_a, "Return handle (link) of a tree root that given link belongs to");
	m.def("find_root_handle", py::overload_cast<const sp_node&>(&find_root_handle),
		"L"_a, "Return handle (link) of a tree root that given node belongs to");
	m.def("convert_path", &convert_path,
		"src_path"_a, "start"_a, "src_path_unit"_a = Key::ID, "dst_path_unit"_a = Key::Name,
		"follow_lazy_links"_a = false,
		"Convert path string from one representation to another (for ex. link IDs -> link names)"
	);
	m.def("deref_path",py::overload_cast<const std::string&, sp_link, Key, bool>(&deref_path),
		"path"_a, "start"_a, "path_unit"_a = Key::ID, "follow_lazy_links"_a = true,
		"Quick link search by given path relative to `start`"
	);
	m.def("deref_path",py::overload_cast<const std::string&, sp_node, Key, bool>(&deref_path),
		"path"_a, "start"_a, "path_unit"_a = Key::ID, "follow_lazy_links"_a = true,
		"Quick link search by given path relative to `start`"
	);
	// async deref_path
	m.def("deref_path",
		py::overload_cast<deref_process_f, std::string, sp_link, Key, bool, bool>(&deref_path),
		"deref_cb"_a, "path"_a, "start"_a, "path_unit"_a = Key::ID,
		"follow_lazy_links"_a = true, "high_priority"_a = false,
		"Async quick link search by given path relative to `start`"
	);

	// bind lists of links & nodes as opaque types
	auto clV = py::bind_vector<links_v>(m, "links_vector", py::module_local(false));
	detail::make_rich_pylist<links_v>(clV);
	bind_list<links_l>(m, "links_list", py::module_local(false));
	bind_list<nodes_l>(m, "nodes_list", py::module_local(false));

	// walk
	// [HINT] pass vectors to be modified as pointers - in this case pybind11 applies reference policy
	using py_walk_links_cb = std::function<void(const sp_link&, links_l*, links_v*)>;
	using py_walk_nodes_cb = std::function<void(const sp_node&, nodes_l*, links_v*)>;

	m.def("walk",
		[](
			const sp_link& root, py_walk_links_cb cb,
			bool topdown, bool follow_symlinks, bool follow_lazy_links
		) {
			py_walk(root, std::move(cb), topdown, follow_symlinks, follow_lazy_links);
		},
		"root"_a, "step_f"_a, "topdown"_a = true, "follow_symlinks"_a = true, "follow_lazy_links"_a = false,
		"Walk the tree similar to Python `os.walk()`"
	);
	m.def("walk",
		[](
			const sp_node& root, py_walk_nodes_cb cb,
			bool topdown, bool follow_symlinks, bool follow_lazy_links
		) {
			py_walk(root, std::move(cb), topdown, follow_symlinks, follow_lazy_links);
		},
		"root_node"_a, "step_f"_a, "topdown"_a = true, "follow_symlinks"_a = true, "follow_lazy_links"_a = false,
		"Walk the tree similar to Python `os.walk()` (alternative)"
	);

	// make root link
	m.def("make_root_link", &make_root_link,
		"link_type"_a = "hard_link", "name"_a = "/", "root_node"_a = nullptr,
		"Make root link pointing to node which handle is preset to returned link"
	);

	// save/load tree
	py::enum_<TreeArchive>(m, "TreeArchive")
		.value("Text", TreeArchive::Text)
		.value("Binary", TreeArchive::Binary)
		.value("FS", TreeArchive::FS)
	;
	m.def("save_tree", py::overload_cast<sp_link, std::string, TreeArchive, timespan>(&save_tree),
		"root"_a, "filename"_a, "ar"_a = TreeArchive::FS, "wait_for"_a = infinite);
	m.def("save_tree", py::overload_cast<on_serialized_f, sp_link, std::string, TreeArchive>(&save_tree),
		"callback"_a, "root"_a, "filename"_a, "ar"_a = TreeArchive::FS);
	m.def("load_tree", py::overload_cast<std::string, TreeArchive>(&load_tree),
		"filename"_a, "ar"_a = TreeArchive::FS);
	m.def("load_tree", py::overload_cast<on_serialized_f, std::string, TreeArchive>(&load_tree),
		"callback"_a, "filename"_a, "ar"_a = TreeArchive::FS);
}

NAMESPACE_END(blue_sky::python)
