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
using Key = node::Key;

/*-----------------------------------------------------------------------------
 *  hidden details
 *-----------------------------------------------------------------------------*/
NAMESPACE_BEGIN()

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

// helpers to omit code duplication
// ------- contains
template<Key K>
bool contains(const node& N, const std::string& key) {
	return N.find(key, K) != N.end<>();
}
bool contains_link(const node& N, const sp_link& l) {
	return N.find(l->id()) != N.end<>();
}
bool contains_obj(const node& N, const sp_obj& obj) {
	return N.find(obj->id(), Key::OID) != N.end<>();
}

// ------- find
template<Key K, bool Throw = true>
auto find(const node& N, const std::string& key) -> sp_link {
	auto r = N.find(key, K);
	if(r != N.end<>()) return *r;
	if(Throw) {
		std::string msg = "Node doesn't contain link ";
		switch(K) {
		default:
		case Key::ID:
			msg += "with ID = "; break;
		case Key::OID:
			msg += "to object with ID = "; break;
		case Key::Name:
			msg += "with name = "; break;
		case Key::Type:
			msg += "with type = "; break;
		}
		throw py::key_error(msg + key);
	}
	return nullptr;
}

template<bool Throw = true>
sp_link find_obj(const node& N, const sp_obj& obj) {
	return find<Key::OID, Throw>(N, obj->id());
}

auto find_by_idx(const node& N, const long idx, bool allow_end = false) {
	// support for Pythonish indexing from both ends
	const std::size_t positive_idx = idx < 0 ? N.size() + std::size_t(allow_end) + idx : std::size_t(idx);
	if(positive_idx > N.size() - std::size_t(!allow_end))
		throw py::key_error("Index out of bounds");
	return std::next(N.begin(), positive_idx);
}

// ------- index
template<Key K>
auto index(const node& N, const std::string& key) -> std::size_t {
	return N.index(key, K);
}
auto index_link(const node& N, const sp_link& l) {
	return N.index(l->id());
}
auto index_obj(const node& N, const sp_obj& obj) {
	return N.index(obj->id(), Key::OID);
}

// ------- deep search
template<Key K>
auto deep_search(const node& N, const std::string& key) -> sp_link {
	return N.deep_search(key, K);
}
auto deep_search_obj(const node& N, const sp_obj& obj) {
	return N.deep_search(obj->id(), Key::OID);
}

// ------- erase
template<Key K>
void erase(node& N, const std::string& key) {
	N.erase(key, K);
}
void erase_link(node& N, const sp_link& l) {
	N.erase(l->id());
}
void erase_obj(node& N, const sp_obj& obj) {
	N.erase(obj->id(), Key::OID);
}
void erase_idx(node& N, const long idx) {
	// support for Pythonish indexing from both ends
	if(std::size_t(std::abs(idx)) > N.size())
		throw py::key_error("Index out of bounds");
	N.erase(idx < 0 ? N.size() + idx : idx);
}

// ------- rename
template<Key K>
int rename(node& N, const std::string& key, std::string new_name, bool all = false) {
	return N.rename(key, std::move(new_name), K, all);
}

NAMESPACE_END() // hidden ns

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

	// `node` binding
	node_pyface
		BSPY_EXPORT_DEF(node)
		.def(py::init<>())
		.def("__len__", &node::size)
		.def("__iter__",
			[](const node& N) { return py::make_iterator(N.begin(), N.end()); },
			py::keep_alive<0, 1>()
		)

		// check by link ID
		.def("__contains__", &contains_link, "link"_a)
		.def("contains",     &contains_link, "link"_a, "Check if node contains given link")
		.def("__contains__", &contains<Key::ID>, "lid"_a)
		.def("contains",     &contains<Key::ID>, "lid"_a, "Check if node contains link with given ID")
		// ... by object
		.def("__contains__", &contains_obj, "obj"_a)
		.def("contains",     &contains_obj, "obj"_a, "Check if node contains link to given object")
		// check by link name
		.def("contains_name", &contains<Key::Name>,
			"link_name"_a, "Check if node contains link with given name")
		// check by object ID
		.def("contains_oid",  &contains<Key::OID>,
			"oid"_a, "Check if node contains link to object with given ID")
		// check by object type
		.def("contains_type", &contains<Key::Type>,
			"obj_type_id"_a, "Check if node contain links to objects of given type")

		// get item by int index
		.def("__getitem__", [](const node& N, const long idx) {
			return *find_by_idx(N, idx);
		}, "link_idx"_a)
		// search by link ID
		.def("__getitem__", &find<Key::ID>, "lid"_a)
		.def("find",        &find<Key::ID, false>, "lid"_a, "Find link with given ID")
		// search by object instance
		.def("__getitem__", &find_obj<>, "obj"_a)
		.def("find",        &find_obj<false>, "obj"_a, "Find link to given object")
		// search by object ID
		.def("find_oid",    &find<Key::OID, false>, "oid"_a, "Find link to object with given ID")
		// search by link name
		.def("find_name",   &find<Key::Name, false>, "link_name"_a, "Find link with given name")

		// obtain index in custom order from link ID
		.def("index", &index<Key::ID>, "lid"_a, "Find index of link with given ID")
		// ...from link itself
		.def("index", &index_link, "link"_a, "Find index of given link")
		// ... from object
		.def("index", &index_obj, "obj"_a, "Find index of link to given object")
		// ... from OID
		.def("index_oid",  &index<Key::OID>, "oid"_a, "Find index of link to object with given ID")
		// ... from link name
		.def("index_name", &index<Key::Name>, "name"_a, "Find index of link with given name")

		// deep search by object
		.def("deep_search", &deep_search_obj, "obj"_a, "Deep search for link to given object")
		// deep search by link ID
		.def("deep_search", &deep_search<Key::ID>, "lid"_a, "Deep search for link with given ID")
		// deep search by link name
		.def("deep_search_name", &deep_search<Key::Name>, "link_name"_a, "Deep search for link with given name")
		// deep search by object ID
		.def("deep_search_oid", &deep_search<Key::OID>, "oid"_a, "Deep search for link to object with given ID")

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
		.def("__delitem__", &erase_idx, "idx"_a)
		.def("erase",       &erase_idx, "idx"_a, "Erase link with given index")
		// erase given link
		.def("__delitem__", &erase_link, "link"_a)
		.def("erase",       &erase_link, "link"_a, "Erase given link")
		// erase by given link ID
		.def("__delitem__", &erase<Key::ID>, "lid"_a)
		.def("erase",       &erase<Key::ID>, "lid"_a, "Erase link with given ID")
		// erase by object instance
		.def("__delitem__", &erase_obj, "obj"_a)
		.def("erase",       &erase_obj, "obj"_a, "Erase links to given object")
		// erase by link name
		.def("erase_name", &erase<Key::Name>, "link_name"_a, "Erase links with given name")
		// erase by OID
		.def("erase_oid",  &erase<Key::OID>, "oid"_a, "Erase links to object with given ID")
		// erase by obj type
		.def("erase_type", &erase<Key::Type>, "obj_type_id"_a, "Erase all links pointing to objects of given type")

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
			return N.rename(lid, std::move(new_name), Key::ID);
		}, "lid"_a, "new_name"_a, "Rename link with given ID")
		// by link instance (extracts ID)
		.def("rename", [](node& N, const sp_link& l, std::string new_name) {
			return N.rename(l->id(), std::move(new_name));
		}, "link"_a, "new_name"_a, "Rename given link")
		// by index offset
		.def("rename", [](node& N, const long idx, std::string new_name) {
			return N.rename(find_by_idx(N, idx), std::move(new_name));
		}, "idx"_a, "new_name"_a, "Rename link with given index")
		// by object instance
		.def("rename", [](node& N, const sp_obj& obj, std::string new_name, bool all = false) {
			return N.rename(obj->id(), std::move(new_name), Key::OID, all);
		}, "obj"_a, "new_name"_a, "all"_a = false, "Rename link(s) to given object")
		// by object ID
		.def("rename_oid", &rename<Key::OID>, "oid"_a, "new_name"_a, "all"_a = false,
			"Rename link(s) to object with given ID")
		.def("rename_name", &rename<Key::Name>, "old_name"_a, "new_name"_a, "all"_a = false,
			"Rename link(s) with given old_name")

		// misc API
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
	m.def("abspath", &abspath, "lnk"_a, "path_unit"_a = node::Key::ID, "Get link's absolute path");
	m.def("convert_path", &convert_path,
		"src_path"_a, "start"_a, "src_path_unit"_a = node::Key::ID, "dst_path_unit"_a = node::Key::Name,
		"Convert path string from one representation to another (for ex. link IDs -> link names)"
	);
	m.def("deref_path", &deref_path, "path"_a, "start"_a, "path_unit"_a = node::Key::ID,
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

