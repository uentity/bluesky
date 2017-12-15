/// @file
/// @author uentity
/// @date 22.09.2017
/// @brief BS tree Python bindings
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/bs.h>
#include <bs/tree.h>
#include <bs/python/tree.h>
#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/string_generator.hpp>
#include <string>

#include <pybind11/functional.h>
#include <iostream>

// make it possible to exchange vector of links without copying
// (and modify it in-place in Python for `walk()`)
PYBIND11_MAKE_OPAQUE(std::vector<blue_sky::tree::sp_link>);
//PYBIND11_MAKE_OPAQUE(std::vector<double>);

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(python)
using namespace tree;

/*-----------------------------------------------------------------------------
 *  hidden details
 *-----------------------------------------------------------------------------*/
namespace {

static boost::uuids::string_generator uuid_from_str;

void print_link(const sp_link& l, int level = 0) {
	static const auto dumplnk = [](const sp_link& l_) {
		std::cout << l_->name() << " [" << l_->type_id() << ' ' << l_->id() << "] -> ("
		          << l_->obj_type_id() << ", " << l_->oid() << ")" << std::endl;
	};

	const std::string loffs(level*2, ' ');
	// print link itself
	std::cout << loffs;
	dumplnk(l);
	if(auto n = l->data_node()) {
		// print leafs
		for(const auto &leaf : *n)
			print_link(leaf, level+1);
	}
}

// impl of some method to omit code duplication
bool check_lid(const node& N, const std::string& lid) {
	return N.find(uuid_from_str(lid)) != N.end<>();
}

auto find_by_lid(const node& N, const std::string& lid) {
	auto r = N.find(uuid_from_str(lid));
	if(r != N.end<>()) return *r;
	throw py::key_error("Node doesn't contain link with ID " + lid);
}

auto find_by_idx(const node& N, const long idx, bool allow_end = false) {
	// support for Pythonish indexing from both ends
	if(std::size_t(std::abs(idx)) > N.size())
		throw py::key_error("Index out of bounds");
	auto pos = idx < 0 ? N.end() : N.begin();
	std::advance(pos, idx);
	if(!allow_end && pos == N.end()) throw py::key_error("Index out of bounds");
	return pos;
}

void erase_by_lid(node& N, const std::string& key) {
	N.erase(uuid_from_str(key));
}

} // hidden ns

/*-----------------------------------------------------------------------------
 *  tree Python bindings
 *-----------------------------------------------------------------------------*/
void py_bind_tree(py::module& m) {
	py::class_<link, py_link<>, std::shared_ptr<link>> link_pyface(m, "link");

	// export Flags enum
	py::enum_<link::Flags>(link_pyface, "Flags", py::arithmetic())
		.value("Persistent", link::Flags::Persistent)
		.value("Disabled", link::Flags::Disabled)
		.export_values()
	;
	py::implicitly_convertible<int, link::Flags>();
	py::implicitly_convertible<long, link::Flags>();

	// link base class
	link_pyface
		.def("clone", &link::clone, "deep"_a = true, "Make shallow or deep copy of link")
		.def("data", &link::data)
		.def("type_id", &link::type_id)
		.def("oid", &link::oid)
		.def("obj_type_id", &link::obj_type_id)
		.def("data_node", &link::data_node)
		.def_property_readonly("id", [](const link& L) {
			return boost::uuids::to_string(L.id());
		})
		.def_property_readonly("name", &link::name)
		.def_property_readonly("owner", &link::owner)
		.def("info", &link::info)
		// here we have to make a copy, no way to pass unique_ptr as param
		.def("set_info", [](py_link<>& l, const inode& i) {
			l.set_info(std::make_unique<inode>(i));
		}, "info"_a)
		.def_property("flags", &link::flags, &link::set_flags)
	;

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

		.def_property_readonly("is_alive", &sym_link::is_alive)
		.def("src_path", &sym_link::src_path, "human_readable"_a = false)
	;

	// forward define node
	py::class_<node, objbase, std::shared_ptr<node>> node_pyface(m, "node");

	// export node's Key enum
	py::enum_<node::Key>(node_pyface, "Key")
		.value("ID", node::Key::ID)
		.value("OID", node::Key::OID)
		.value("Name", node::Key::Name)
		.value("Type", node::Key::Type)
		.value("AnyOrder", node::Key::AnyOrder)
	;
	// export node's insert policy
	py::enum_<node::InsertPolicy>(node_pyface, "InsertPolicy", py::arithmetic())
		.value("AllowDupNames", node::InsertPolicy::AllowDupNames)
		.value("DenyDupNames", node::InsertPolicy::DenyDupNames)
		.value("RenameDup", node::InsertPolicy::RenameDup)
		.value("DenyDupOID", node::InsertPolicy::DenyDupOID)
		.value("Merge", node::InsertPolicy::Merge)
		.export_values();
	;
	py::implicitly_convertible<int, node::InsertPolicy>();
	py::implicitly_convertible<long, node::InsertPolicy>();

	// fill node with methods
	node_pyface
		BSPY_EXPORT_DEF(node)
		.def(py::init<>())
		.def("__len__", &node::size)
		.def("__iter__",
			[](const node& N) { return py::make_iterator(N.begin(), N.end()); },
			py::keep_alive<0, 1>()
		)

		// check by link ID
		.def("__contains__", check_lid, "lid"_a)
		// and by object object's ID
		.def("__contains__", [](const node& N, const sp_obj key) {
			return N.find_oid(key->id()) != N.end<>();
		}, "obj"_a)
		// check by link ID
		.def("contains", check_lid, "lid"_a, "Check if node contains link with given ID")
		// check by link name
		.def("contains_name", [](const node& N, const std::string& lname) {
			return N.find(lname) != N.end<>();
		}, "link_name"_a, "Check if node contains link with given name")
		// check by object ID
		.def("contains_oid", [](const node& N, const std::string& oid) {
			return N.find_oid(oid) != N.end<>();
		}, "oid"_a, "Check if node contains link to object with given ID")
		// check by object type
		.def("contains_type", [](const node& N, const std::string& type_id) {
			return N.equal_type(type_id).first != N.end<node::Key::Type>();
		}, "obj_type_id"_a, "Check if node contain links to objects of given type")

		// get item by int index
		.def("__getitem__", [](const node& N, const long idx) {
			return *find_by_idx(N, idx);
		}, "link_idx"_a)
		// search by link ID
		.def("__getitem__", find_by_lid, "lid"_a)
		// search by object instance
		.def("__getitem__", [](const node& N, const sp_obj& obj) {
			auto r = N.find(obj->id());
			if(r != N.end<>()) return *r;
			throw py::key_error("Node doesn't contain links to object with ID = " + obj->id());
		}, "obj"_a)
		// search by link ID
		.def("find", find_by_lid, "lid"_a, "Find link with given ID")
		// search by object ID
		.def("find_oid", [](const node& N, const std::string& oid) -> sp_link {
			auto r = N.find_oid(oid);
			if(r != N.end<>()) return *r;
			return nullptr;
			//throw py::key_error("Node doesn't contain links to object with ID - " + oid);
		}, "oid"_a, "Find link to object with given ID")
		// search by link name
		.def("find_name", [](const node& N, const std::string& lname) -> sp_link {
			auto r = N.find(lname);
			if(r != N.end<>()) return *r;
			return nullptr;
			//throw py::key_error("Node doesn't contain link with name '" + lname + "'");
		}, "link_name"_a, "Find link with given name")

		// obtain index in custom order from link
		.def("index", [](const node& N, const std::string& lid) {
			return N.index(uuid_from_str(lid));
		}, "lid"_a, "Find index of link with given ID")
		.def("index", [](const node& N, const sp_link& l) {
			return N.index(l->id());
		}, "link"_a, "Find index of given link")

		// deep search by link ID or object ID
		.def("deep_search", [](const node& N, const std::string lid) {
			return N.deep_search(uuid_from_str(lid));
		}, "lid"_a, "Deep search for link with given ID")
		.def("deep_search_name", [](const node& N, const std::string& lname) {
			return N.deep_search(lname);
		}, "link_name"_a, "Deep search for link with given name")
		// deep search by object ID
		.def("deep_search_oid", [](const node& N, const std::string& oid) {
			return N.deep_search_oid(oid);
		}, "oid"_a, "Deep search for link to object with given ID")

		.def("equal_range", [](const node& N, const std::string& link_name) {
			auto r = N.equal_range(link_name);
			return py::make_iterator(r.first, r.second);
		}, py::keep_alive<0, 1>(), "link_name"_a)
		.def("equal_range_oid", [](const node& N, const std::string& oid) {
			auto r = N.equal_range_oid(oid);
			return py::make_iterator(r.first, r.second);
		}, py::keep_alive<0, 1>(), "OID"_a)
		.def("equal_type", [](const node& N, const std::string& type_id) {
			auto r = N.equal_type(type_id);
			return py::make_iterator(r.first, r.second);
		}, py::keep_alive<0, 1>(), "obj_type_id"_a)

		// insert given link
		.def("insert", [](node& N, const sp_link& l, node::InsertPolicy pol = node::AllowDupNames) {
			return N.insert(l, uint(pol)).second;
		}, "link"_a, "pol"_a = node::AllowDupNames, "Insert given link")
		// insert link at given index
		.def("insert", [](node& N, const sp_link& l, const long idx, node::InsertPolicy pol = node::AllowDupNames) {
			return N.insert(l, find_by_idx(N, idx, true), pol).second;
		}, "link"_a, "idx"_a, "pol"_a = node::AllowDupNames, "Insert link at given index")
		// insert hard link to given object
		.def("insert", [](node& N, std::string name, sp_obj obj, node::InsertPolicy pol = node::AllowDupNames) {
			return N.insert(std::move(name), std::move(obj), pol).second;
		}, "name"_a, "obj"_a, "pol"_a = node::AllowDupNames, "Insert hard link to given object")

		// erase by given index
		.def("__delitem__", [](node& N, const long idx) {
			// support for Pythonish indexing from both ends
			if(std::size_t(std::abs(idx)) > N.size())
				throw py::key_error("Index out of bounds");
			N.erase(idx < 0 ? N.size() + idx : idx);
		})
		// erase by given link ID
		.def("__delitem__", erase_by_lid, "lid"_a)
		.def("__delitem__", [](node& N, const sp_obj& key) {
			N.erase_oid(key->id());
		}, "obj"_a)
		.def("erase", erase_by_lid, "lid"_a, "Erase link with given ID")
		.def("erase_name", (void (node::*)(const std::string&))&node::erase,
			"link_name"_a, "Erase links with given name")
		.def("erase_oid", &node::erase_oid, "obj"_a, "Erase links to given object")
		.def("erase_type", [](node& N, const std::string& type_id) {
			N.erase_type(type_id);
		}, "obj_type_id"_a, "Erase all links pointing to objects of given type")

		// misc container-related functions
		.def_property_readonly("size", &node::size)
		.def_property_readonly("empty", &node::empty)
		.def("clear", &node::clear, "Clears all node contents")
		.def("keys", [](const node& N, node::Key ktype = node::Key::ID) {
			if(ktype == node::Key::ID) {
				// convert UUIDs to string representation
				auto keys = N.keys<>();
				std::vector<std::string> res;
				res.reserve(keys.size());
				for(const auto& k : keys)
					res.emplace_back(boost::uuids::to_string(k));
				return res;
			}
			else {
				switch(ktype) {
				default:
				case node::Key::Name :
					return N.keys<node::Key::Name>();
				case node::Key::OID :
					return N.keys<node::Key::OID>();
				case node::Key::Type :
					return N.keys<node::Key::Type>();
				};
			}
		}, "key_type"_a = node::Key::ID)

		// link rename
		// by ID
		.def("rename", [](node& N, const std::string& lid, std::string new_name) {
			return N.rename(N.find(uuid_from_str(lid)), std::move(new_name));
		}, "LID"_a, "new_name"_a, "Rename link with given ID")
		// by link instance (extracts ID)
		.def("rename", [](node& N, const sp_link& l, std::string new_name) {
			return N.rename(N.find(l->id()), std::move(new_name));
		}, "link"_a, "new_name"_a, "Rename given link")
		.def("rename", [](node& N, const long idx, std::string new_name) {
			return N.rename(find_by_idx(N, idx), std::move(new_name));
		}, "idx"_a, "new_name"_a, "Rename link with given index")
		.def("rename_all", [](node& N, const std::string& old_name, const std::string& new_name) {
			const auto R = N.equal_range(old_name);
			for(auto p = R.begin(); p != R.end(); ++p)
				N.rename(N.project(p), new_name);
		}, "old_name"_a, "new_name"_a, "Rename all links with given names")

		.def("accepts", &node::accepts, "Check if node accepts given link")
		.def("accept_object_types", &node::accept_object_types,
			"Set white list of object types that node will accept"
		)
		.def_property_readonly("allowed_object_types", &node::allowed_object_types,
			"Returns white list of object types that node will accept"
		)

		.def_property_readonly("handle", &node::handle,
			"Returns a single link that owns this node in overall tree"
		)
		.def("propagate_owner", &node::propagate_owner, "deep"_a = false,
			"Set owner of all contained links to this node (if deep, fix owner in entire subtree)"
		)
	;

	// misc tree-related functions
	m.def("abspath", &abspath, "lnk"_a, "human_readable"_a = false, "Get link's absolute path");
	m.def("convert_path", &convert_path,
		"src_path"_a, "start"_a, "from_human_readable"_a = false,
		"Convert path from ID-based to human readable (name-bases) or vice versa"
	);
	m.def("deref_path", &deref_path, "path"_a, "start"_a,
		"Quick link search by given abs path (ID-based!)"
	);

	// bind list of links as opaque type to allow in-place modification
	py::bind_vector<std::vector<sp_link>>(m, "links_list");
	m.def("walk",
		py::overload_cast<const sp_link&, const step_process_f&, bool, bool>(&walk),
		"root"_a, "step_f"_a, "topdown"_a = true, "follow_symlinks"_a = true,
		"Walk the tree similar to Python `os.walk()`"
	);

	// TEST CODE
	//py::bind_vector<std::vector<double>>(m, "d_list", py::module_local(false));
	//m.def("test_callback", [](std::function<void(std::vector<double>&)> f) {
	//	//std::vector<sp_link> v{nullptr, nullptr, nullptr};
	//	std::vector<double> v{0, 0};
	//	f(v);
	//	for(const auto& l : v) {
	//		std::cout << l << ' ';
	//	}
	//	std::cout << std::endl;
	//});
	//m.def("test_static_d", []() -> std::vector<double>& {
	//	static std::vector<double> v{0, 0};
	//	return v;
	//}, py::return_value_policy::reference);

	m.def("print_link", [](const sp_link& l) { print_link(l, 0); });
	m.def("print_link", [](const sp_node& n, std::string name = "/") {
		print_link(std::make_shared<tree::hard_link>(name, n), 0);
	}, "node"_a, "root_name"_a = "/");
}

NAMESPACE_END(python)
NAMESPACE_END(blue_sky)

