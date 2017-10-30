/// @file
/// @author uentity
/// @date 22.09.2017
/// @brief BS tree Python bindings
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/bs.h>
#include <bs/link.h>
#include <bs/node.h>
#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/string_generator.hpp>
#include <string>

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(python)
using namespace tree;

namespace {

template<typename Link = link>
class py_link : public Link {
public:
	using Link::Link;

	sp_link clone() const override {
		PYBIND11_OVERLOAD_PURE(sp_link, Link, clone, );
	}

	sp_obj data() const override {
		PYBIND11_OVERLOAD_PURE(sp_obj, Link, data, );
	}

	std::string type_id() const override {
		PYBIND11_OVERLOAD_PURE(std::string, Link, type_id, );
	}

	std::string oid() const override {
		PYBIND11_OVERLOAD_PURE(std::string, Link, oid, );
	}

	std::string obj_type_id() const override {
		PYBIND11_OVERLOAD_PURE(std::string, Link, obj_type_id, );
	}

	sp_node data_node() const override {
		PYBIND11_OVERLOAD_PURE(sp_node, Link, data_node, );
	}
};

static boost::uuids::string_generator uuid_from_str;

}

void py_bind_tree(py::module& m) {
	py::class_<inode>(m, "inode")
		.def(py::init<std::string, std::string>())
		.def_readonly("owner", &inode::owner)
		.def_readonly("group", &inode::group)
		.def_property_readonly("suid", [](const inode& i) { return i.suid; })
		.def_property_readonly("sgid", [](const inode& i) { return i.sgid; })
		.def_property_readonly("sticky", [](const inode& i) { return i.sticky; })
		.def_property_readonly("u", [](const inode& i) { return i.u; })
		.def_property_readonly("g", [](const inode& i) { return i.g; })
		.def_property_readonly("o", [](const inode& i) { return i.o; })
	;

	py::class_<link, py_link<>, std::shared_ptr<link>> link_pyface(m, "link");
	link_pyface
		.def(py::init<std::string>())
		.def("clone", &link::clone)
		.def("data", &link::data)
		.def("type_id", &link::type_id)
		.def("oid", &link::oid)
		.def("obj_type_id", &link::obj_type_id)
		.def("data_node", &link::data_node)
		.def_property_readonly("id", [](const link& L) {
			return boost::uuids::to_string(L.id());
		})
		.def_property_readonly("name", &link::name)
		.def_property("inode",
			py::overload_cast<>(&link::get_inode, py::const_),
			py::overload_cast<>(&link::get_inode)
		)
		.def_property_readonly("owner", &link::owner)
	;

	py::class_<hard_link, link, py_link<hard_link>, std::shared_ptr<hard_link>>(m, "hard_link")
		.def(py::init<std::string, sp_obj>())
	;
	py::class_<weak_link, link, py_link<weak_link>, std::shared_ptr<weak_link>>(m, "weak_link")
		.def(py::init<std::string, const sp_obj&>())
	;

	// forward define node
	py::class_<node, objbase, std::shared_ptr<node>> node_pyface(m, "node");

	// export node's Key enum
	py::enum_<node::Key>(node_pyface, "Key")
		.value("ID", node::Key::ID)
		.value("OID", node::Key::OID)
		.value("Name", node::Key::Name)
		.value("Type", node::Key::Type)
	;
	// export node's insert policy
	py::enum_<node::InsertPolicy>(node_pyface, "InsertPolicy")
		.value("AllowDupNames", node::InsertPolicy::AllowDupNames)
		.value("DenyDupNames", node::InsertPolicy::DenyDupNames)
		.value("RenameDup", node::InsertPolicy::RenameDup)
		.value("DenyDupOID", node::InsertPolicy::DenyDupOID)
		.value("Merge", node::InsertPolicy::Merge)
		.export_values();
	;

	// fill node with methods
	node_pyface
		BSPY_EXPORT_DEF(node)
		.def(py::init<>())
		.def("__len__", &node::size)
		.def("__iter__",
			[](const node& N) { return py::make_iterator(N.begin(), N.end()); },
			py::keep_alive<0, 1>()
		)

		// check by link name or object instance (object's ID)
		.def("__contains__", [](const node& N, py::object key) {
			if(PyString_Check(key.ptr()))
				return N.find(py::cast<std::string>(key)) != N.end<>();
			return N.find_oid(py::cast<sp_obj>(key)->id()) != N.end<>();
		})
		// check by link ID
		.def("contains_lid", [](const node& N, const std::string& lid) {
			return N.find(uuid_from_str(lid)) != N.end<>();
		}, "LID", "Check if node contains link with given ID")
		// check by object ID
		.def("contains_oid", [](const node& N, const std::string& oid) {
			return N.find_oid(oid) != N.end<>();
		}, "OID", "Check if node contains object with given ID")
		// check by object type
		.def("contains_type", [](const node& N, const std::string& type_id) {
			return N.equal_type(type_id).first != N.end<node::Key::Type>();
		}, "obj_type_id", "Check if node contains objects with given type")

		// search by link name
		.def("__getitem__", [](const node& N, const std::string& link_name) {
			auto r = N.find(link_name);
			if(r != N.end<>()) return *r;
			throw py::key_error("Node doesn't contain link with name '" + link_name + "'");
		}, py::return_value_policy::reference_internal, "link_name"_a)
		// search by object instance
		.def("__getitem__", [](const node& N, const sp_obj& obj) {
			auto r = N.find(obj->id());
			if(r != N.end<>()) return *r;
			throw py::key_error("Node doesn't contain object with ID = " + obj->id());
		}, py::return_value_policy::reference_internal, "object"_a)
		// get item by int index
		.def("__getitem__", [](const node& N, long idx) {
			if(idx < 0 || ulong(idx) >= N.size())
				throw py::key_error("Index out of bounds");
			auto r = N.begin();
			std::advance(r, idx);
			return *r;
		}, py::return_value_policy::reference_internal, "link_name"_a)
		// search by object ID
		.def("find_oid", [](const node& N, const std::string& oid) {
			auto r = N.find_oid(oid);
			if(r != N.end<>()) return *r;
			throw py::key_error("Node doesn't contain object with ID - " + oid);
		}, py::return_value_policy::reference_internal, "OID"_a, "Find item by object ID")
		// search by link ID
		.def("find_lid", [](const node& N, const std::string& lid) {
			auto r = N.find(uuid_from_str(lid));
			if(r != N.end<>()) return *r;
			throw py::key_error("Node doesn't contain link with ID - " + lid);
		}, py::return_value_policy::reference_internal, "LID"_a, "Find item by object link ID")

		// deep search by link name or object instance (OID)
		.def("deep_search", [](const node& N, py::object key) {
			if(PyString_Check(key.ptr()))
				return N.deep_search(py::cast<std::string>(key));
			return N.deep_search_oid(py::cast<sp_obj>(key)->id());
		}, "key"_a, "Deep search for link with given name or by object instance (OID)")
		.def("deep_search_lid", [](const node& N, const std::string& lid) {
			return N.deep_search(uuid_from_str(lid));
		}, "LID"_a, "Deep search for link with given ID")
		// deep search by object ID
		.def("deep_search_oid", [](const node& N, const std::string& oid) {
			return N.deep_search_oid(oid);
		}, "OID"_a, "Deep search for object with given ID")

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

		// insert hard link to given object
		.def("__setitem__", [](node& N, std::string link_name, sp_obj obj) {
			N.insert(std::move(link_name), std::move(obj));
		}, "link_name"_a, "obj"_a)
		// insert given link
		.def("insert", [](node& N, sp_link l, uint pol = node::AllowDupNames) {
			return N.insert(std::move(l)).second;
		}, "link"_a, "pol"_a = node::AllowDupNames, "Insert given link")
		// insert hard link to given object
		.def("insert", [](node& N, std::string name, sp_obj obj, uint pol = node::AllowDupNames) {
			return N.insert(std::move(name), std::move(obj), pol).second;
		}, "name"_a, "obj"_a, "pol"_a = node::AllowDupNames, "Insert hard link to given object")

		// erase by given link name or object instance (object's ID)
		.def("__delitem__", [](node& N, py::object key) {
			if(PyString_Check(key.ptr()))
				N.erase(py::cast<std::string>(key));
			N.erase_oid(py::cast<sp_obj>(key)->id());
		})
		.def("erase", (void (node::*)(const std::string&))&node::erase,
			"link_name"_a, "Erase links with given name")
		.def("erase_oid", &node::erase_oid, "obj"_a, "Erase links to given object")
		.def("erase_lid", [](node& N, const std::string& key) {
			N.erase(uuid_from_str(key));
		}, "lid"_a, "Erase link with given ID")
		.def("erase_type", [](node& N, const std::string& type_id) {
			N.erase_type(type_id);
		}, "obj_type_id"_a, "Erase all links pointing to objects with given type")

		// misc functions
		.def_property_readonly("size", &node::size)
		.def_property_readonly("empty", &node::empty)
		.def("clear", &node::clear, "Clears all node contents")
		.def("keys", [](const node& N, node::Key ktype = node::Key::Name) {
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
		}, "key_type"_a = node::Key::Name)
	;
}

NAMESPACE_END(python)
NAMESPACE_END(blue_sky)

