/// @file
/// @author uentity
/// @date 22.09.2017
/// @brief BS tree Python bindings
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/python/common.h>
#include <bs/python/expected.h>
#include <bs/python/list.h>
#include <bs/tree/tree.h>

#include <pybind11/functional.h>
#include <pybind11/chrono.h>

// make it possible to bind opaque std::list & std::vector (w/o content copying)
PYBIND11_MAKE_OPAQUE(std::vector<blue_sky::tree::sp_link>);
PYBIND11_MAKE_OPAQUE(std::list<blue_sky::tree::sp_link>);

NAMESPACE_BEGIN(blue_sky::python)
using namespace tree;
using Key = node::Key;

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

	// bind list of links as opaque type 
	using v_links = std::vector<sp_link>;
	using l_links = std::list<sp_link>;
	auto clV = py::bind_vector<v_links>(m, "links_vector", py::module_local(false));
	detail::make_rich_pylist<v_links>(clV);
	bind_list<l_links>(m, "links_list", py::module_local(false));

	// walk
	// [NOTE] use proxy functor to allow nodes list modification in Python callback
	// [HINT] pass vectors to be modified as pointers - in this case pybind11 applies reference policy
	using py_walk_cb = std::function<void(const sp_link&, l_links*, v_links*)>;
	m.def("walk",
		[](
			const sp_link& root, py_walk_cb pyf, bool topdown = true, bool follow_symlinks = true,
			bool follow_lazy_links = false
		) {
			walk(root, [pyf = std::move(pyf)] (
					const sp_link& cur_root, l_links& nodes, v_links& leafs
				) {
					// convert references to pointers
					pyf(cur_root, &nodes, &leafs);
				},
				topdown, follow_symlinks
			);
		},
		"root"_a, "step_f"_a, "topdown"_a = true, "follow_symlinks"_a = true, "follow_lazy_links"_a = false,
		"Walk the tree similar to Python `os.walk()`"
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
	;
	m.def("save_tree", &save_tree, "root"_a, "filename"_a, "ar"_a = TreeArchive::Text);
	m.def("load_tree", &load_tree, "filename"_a, "ar"_a = TreeArchive::Text);
}

NAMESPACE_END(blue_sky::python)
