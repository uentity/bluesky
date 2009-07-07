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

#include "bs_tree.h"
#include "py_bs_object_base.h"
#include "py_bs_link.h"
#include "py_bs_tree.h"
#include "bs_shell.h"
#include "py_bs_exports.h"

#include <string>

using namespace std;

namespace blue_sky {
namespace python {
//py_bs_inode::py_bs_inode() : spinode(NULL) {}

py_bs_inode::py_bs_inode(const sp_inode &tsp) : spinode(tsp) {}

py_objbase* py_bs_inode::data() const {
	 return new py_objbase(spinode->data());
}

ulong py_bs_inode::size() const {
	 return spinode->size();
}

uint py_bs_inode::uid() const {
	 return spinode->uid();
}

uint py_bs_inode::gid() const {
	 return spinode->gid();
}

uint py_bs_inode::mode() const {
	 return spinode->mode();
}

time_t py_bs_inode::mtime() const {
	 return spinode->mtime();
}

py_bs_link::py_bs_link(const sp_link &tsp) : splink(tsp)
{}

py_bs_link::py_bs_link(const py_bs_link &tsp) : splink(tsp.splink)
{}

/*py_bs_link::py_bs_link(const py_objbase& obj, const py_bs_link& root, bool is_persistent)
 : splink(new bs_link(obj.sp,root.splink,is_persistent))
{}*/

py_bs_inode py_bs_link::inode() const {
	 return py_bs_inode(splink->inode());
}

py_objbase py_bs_link::data() const {
	 return py_objbase(splink->data());
}

std::string py_bs_link::name() const {
	 return splink->name();
}

std::string py_bs_link::full_name() const {
 //deep_iterator di(sp_shell(splink->data()));
 //for bs_node::n_iterator iter = bs_node::begin();
 return "Fuck";//di.full_name();
}

bool py_bs_link::is_node() const {
	 return splink->is_node();
}

py_bs_node *py_bs_link::node() const {
 if (splink->is_node())
	 return new py_bs_node(splink->node());
 return NULL;
}

/*bool py_bs_link::is_soft() const {
	 return splink->is_soft();
}*/

//bool py_bs_link::is_persistent() const {
//	 return splink->is_persistent();
//}

bool py_bs_link::is_hard_link() const {
 return splink->is_hard_link();
}

bs_link::link_type py_bs_link::link_type_id() const {
 return splink->link_type_id();
}

py_bs_link py_bs_link::clone(const std::string& clone_name) const {
 return py_bs_link(splink->clone(clone_name));
}

py_bs_link py_bs_link::dumb_link(const std::string name) {
 return py_bs_link(bs_link::dumb_link(name));
}

/*py_bs_link py_bs_link::copy(const py_bs_link& where) {
	 return splink.lock()->copy(where.splink);
}

bool py_bs_link::move(const py_bs_link& where) {
	 return splink.lock()->move(where.splink);
}

py_bs_link py_bs_link::soft_clone() const {
	 return splink->soft_clone();
}*/

void py_export_link() {
	class_< py_bs_inode >("inode",init <const py_bs_inode::sp_inode&>())
		.def("data",&py_bs_inode::data,return_value_policy<manage_new_object>())
		.def("size",&py_bs_inode::size)
		.def("uid",&py_bs_inode::uid)
		.def("gid",&py_bs_inode::gid)
		.def("mode",&py_bs_inode::mode)
		.def("mtime",&py_bs_inode::mtime);

	class_< py_bs_link >("link",init <const py_bs_link&>())
		.def("inode",&py_bs_link::inode)
		.def("data",&py_bs_link::data)
		.def("name",&py_bs_link::name)
		.def("full_name",&py_bs_link::full_name)
		.def("is_node",&py_bs_link::is_node)
		.def("node",&py_bs_link::node,return_value_policy<manage_new_object>());
		//.def("is_persistent",&py_bs_link::is_persistent);

	class_< std::list< py_bs_link > >("list_link")
		.def("__iter__",boost::python::iterator< std::list< py_bs_link > >());
}

}	//namespace blue_sky::python
}	//namespace blue_sky
