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

/*!
 * \file bs_object_base.cpp
 * \brief Just blue-sky object base implimentations.
 * \author Gagarin Alexander <gagrinav@ufanipi.ru>
 * \date 2007-03-05
 */

#include "bs_object_base.h"
#include "bs_command.h"
#include "bs_tree.h"
#include "bs_kernel.h"
#include "bs_prop_base.h"

using namespace std;
using namespace blue_sky;
using namespace Loki;

// -----------------------------------------------------
// Implementation of class: object_base
// -----------------------------------------------------

//namespace blue_sky {
//	namespace bs_private {
//		bs_mutex objbase_name_guard;
//	}
//}

BS_COMMON_IMPL(objbase)

//bool objbase::bs_register_instance(const smart_ptr< objbase, true >& sp_inst) {
//	lsmart_ptr< sp_objinst > l_inst(bs_instances());
//	if(find(l_inst->begin(), l_inst->end(), sp_inst) == l_inst->end()) {
//		l_inst->push_back(sp_inst);
//		return true;
//	}
//	else return false;
//}
//
//void objbase::bs_free_instance(const smart_ptr< objbase, true >& sp_inst) {
//	bs_instances().lock()->remove(sp_inst);
//}

/*! WARNING!
	All objbase's ctors sets reference counter = 1 in order to allow
	using smart_ptr(this) in constructors of inherited classes
	refcounter is decremented to correct value in kernel.create_object and create_object_copy methods
*/

/*!
 * \brief Default constructor
 */
objbase::objbase(bs_type_ctor_param)
	: bs_refcounter(), bs_messaging(BS_SIGNAL_RANGE(objbase)) //, inode_(new blue_sky::bs_inode(this))
{
	add_ref();
}

objbase::objbase(const bs_messaging::sig_range_t& sr)
	: bs_refcounter(), bs_messaging(sr) //, inode_(new blue_sky::bs_inode(this))
{
	add_ref();
	add_signal(BS_SIGNAL_RANGE(objbase));
}

objbase::objbase(const objbase& obj)
	: bs_refcounter(), bs_messaging(obj) //, inode_(new blue_sky::bs_inode(this))
{
	add_ref();
}

void objbase::swap(objbase& rhs) {
	//swap signals
	bs_messaging::swap(rhs);
	std::swap(inode_, rhs.inode_);
}

//objbase& objbase::operator =(const objbase& obj)
//{
//	//try to make copy of given object
//	sp_obj tmp = BS_KERNEL.create_object_copy(&obj, true);
//	if(tmp) {
//		//copy creation successful - swap this and copy
//		tmp.lock()->swap(*this);
//	}
//	else {
//		objbase::swap(objbase(obj), *this);
//	}
//}

/*!
 * \brief Destructor
 */
objbase::~objbase()
{}

void objbase::dispose() const {
	delete this;
}

smart_ptr< bs_inode, true > objbase::inode() const {
	return inode_;
}

type_descriptor objbase::bs_type() {
	return type_descriptor(BS_GET_TI(objbase), NULL, NULL, NULL, "blue_sky::objbase", "Base class of \
		all BlueSky types");
}

int objbase::bs_register_this() const {
	return BS_KERNEL.register_instance(this);
}

int objbase::bs_free_this() const {
	return BS_KERNEL.free_instance(this);
}

// -----------------------------------------------------
// End of implementation class: object_base
// -----------------------------------------------------
