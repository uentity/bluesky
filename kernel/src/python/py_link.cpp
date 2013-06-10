// This file is part of BlueSky
// 
// BlueSky is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.
// 
// BlueSky is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with BlueSky; if not, see <http://www.gnu.org/licenses/>.

#include "bs_link.h"
#include "bs_tree.h"
#include "py_bs_exports.h"
#include "py_smart_ptr.h"

namespace blue_sky { namespace python {

using namespace boost::python;

namespace {
using namespace std;
// objbase wrapper
class bs_link_pyw : public bs_link, public wrapper< bs_link > {
public:
	string name() const {
		if(override f = this->get_override("name"))
			return (const string&)f();
		else
			return bs_link::name();
	}

	string default_name() const {
		return bs_link::name();
	}

	bool is_node() const {
		if(override f = this->get_override("is_node"))
			return f();
		else
			return bs_link::is_node();
	}

	bool default_is_node() const {
		return bs_link::is_node();
	}

	link_type link_type_id() const {
		if(override f = this->get_override("link_type_id"))
			return f();
		else
			return bs_link::link_type_id();
	}

	link_type default_link_type_id() const {
		return bs_link::link_type_id();
	}

	sp_link clone(const std::string& clone_name = "") const {
		if(override f = this->get_override("clone"))
			return f(clone_name);
		else
			return bs_link::clone(clone_name);
	}

	sp_link default_clone(const std::string& clone_name = "") const {
		return bs_link::clone(clone_name);
	}

	sp_node parent() const {
		if(override f = this->get_override("parent"))
			return f();
		else
			return bs_link::parent();
	}
	sp_node def_parent() const {
		return bs_link::parent();
	}
};

// link hash is it's address - so multiple smart_ptrs poining to the same link will have
// identical hashes
std::size_t link_hash(const bs_link& l) {
	return reinterpret_cast< std::size_t >(&l);
	//return uint(big_hash);
}

}

// exporting function
void py_bind_link() {
	// bs_inode binding
	class_<
		bs_inode,
		smart_ptr< bs_inode, true >,
		bases< bs_refcounter >,
		boost::noncopyable
		>
	("inode", no_init)
		.add_property("size", &bs_inode::size)
		.add_property("uid", &bs_inode::uid)
		.add_property("gid", &bs_inode::gid)
		.add_property("mode", &bs_inode::mode)
		.add_property("mtime", &bs_inode::mtime)
		.def("links_begin", &bs_inode::links_begin)
		.def("links_end", &bs_inode::links_end)
		.add_property("links_count", &bs_inode::links_count)
		.add_property("data", &bs_inode::data)
		;

	// bs_link binding
	class_<
		bs_link_pyw,
		smart_ptr< bs_link_pyw, true >,
		bases< objbase >,
		boost::noncopyable
		>
	("link", no_init)
		.def("__hash__", &link_hash)
		.def("create", &bs_link::create)
		.staticmethod("create")
		.def("bs_type", &bs_link::bs_type)
		.staticmethod("bs_type")
		.def("bs_resolve_type", &bs_link::bs_resolve_type)
		.add_property("inode", &bs_link::inode)
		.add_property("data", &bs_link::data)
		.def("name", &bs_link::name, &bs_link_pyw::default_name)
		.def("is_node", &bs_link::is_node, &bs_link_pyw::default_is_node)
		.add_property("node", &bs_link::node)
		.def("is_hard_link", &bs_link::is_hard_link)
		.def("link_type_id", &bs_link::link_type_id, &bs_link_pyw::default_link_type_id)
		.def("clone", &bs_link::clone, &bs_link_pyw::default_clone)
		.def("dumb_link", &bs_link::dumb_link)
		.def("parent", &bs_link::parent, &bs_link_pyw::def_parent)
		;

	// register smart_ptr conversions
	implicitly_convertible< smart_ptr< bs_link_pyw, true >, blue_sky::smart_ptr< bs_link, true > >();
	register_ptr_to_python< blue_sky::smart_ptr< bs_link, true > >();

	// bs_alias bindings
	class_<
		bs_alias,
		smart_ptr< bs_alias >,
		bases< bs_link >,
		boost::noncopyable
		>
	("alias", no_init)
		.def("create", &bs_alias::create)
		.staticmethod("create")
		.def("bs_type", &bs_alias::bs_type)
		.staticmethod("bs_type")
		.def("bs_resolve_type", &bs_alias::bs_resolve_type)
		.def("clone", &bs_alias::clone)
		.def("link_type_id", &bs_alias::link_type_id)
		;
}

}}	// eof namespace blue_sky::python

