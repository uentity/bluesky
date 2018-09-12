/// @file
/// @author uentity
/// @date 16.11.2017
/// @brief Trampolines and other public stuff for tree Python bindings
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <pybind11/pybind11.h>
#include <bs/tree/link.h>
#include <bs/tree/node.h>

NAMESPACE_BEGIN(blue_sky) NAMESPACE_BEGIN(python)

/*-----------------------------------------------------------------------------
 *  link's trampoline class for derived types
 *-----------------------------------------------------------------------------*/
template<typename Link = tree::link>
class py_link : public Link {
public:
	using Link::Link;

	tree::sp_link clone(bool deep = false) const override {
		PYBIND11_OVERLOAD_PURE(tree::sp_link, Link, clone, deep);
	}

	std::string type_id() const override {
		PYBIND11_OVERLOAD_PURE(std::string, Link, type_id, );
	}

	std::string oid() const override {
		PYBIND11_OVERLOAD(std::string, Link, oid, );
	}

	std::string obj_type_id() const override {
		PYBIND11_OVERLOAD(std::string, Link, obj_type_id, );
	}

	result_or_err<sp_obj> data_impl() const override {
		PYBIND11_OVERLOAD_PURE(sp_obj, Link, data_impl, );
	}

	result_or_err<tree::sp_node> data_node_impl() const override {
		PYBIND11_OVERLOAD(tree::sp_node, Link, data_node_impl, );
	}

	//typename Link::Flags flags() const override {
	//	PYBIND11_OVERLOAD(typename Link::Flags, Link, flags, );
	//}
	//void set_flags(typename Link::Flags new_flags) override {
	//	PYBIND11_OVERLOAD(void, Link, set_flags, new_flags);
	//}
};

/*-----------------------------------------------------------------------------
 *  node's trampoline class for derived types
 *-----------------------------------------------------------------------------*/
// [NOTE] disabled, because there's no virtual methods in node
//template<typename Node = tree::node>
//class py_node : public Node {
//public:
//	using Node::Node;
//	using typename Node::InsertPolicy;
//
//	tree::sp_node deep_clone(InsertPolicy pol = InsertPolicy::AllowDupNames) const override {
//		PYBIND11_OVERLOAD(tree::sp_node, Node, deep_clone, pol);
//	}
//};

NAMESPACE_END(python) NAMESPACE_END(blue_sky)

