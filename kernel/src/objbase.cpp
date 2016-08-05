/// @file
/// @author uentity
/// @date 05.03.2007
/// @brief Just BlueSky object base implimentation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/objbase.h>
//#include "bs_command.h"
//#include "bs_tree.h"
//#include "bs_kernel.h"
//#include "bs_prop_base.h"


// -----------------------------------------------------
// Implementation of class: object_base
// -----------------------------------------------------

//BS_COMMON_IMPL(objbase)

namespace blue_sky {

/*!
 * \brief Default constructor
 */

objbase::objbase(bs_type_ctor_param)
	:  inode_(NULL)
{}

//objbase::objbase(const bs_messaging::sig_range_t& sr)
//	: bs_refcounter(), bs_messaging(sr), inode_(NULL)
//{
//	add_ref();
//	add_signal(BS_SIGNAL_RANGE(objbase));
//}

objbase::objbase(const objbase& obj)
	: inode_(NULL)
{}

void objbase::swap(objbase& rhs) {
	//bs_messaging::swap(rhs);
	std::swap(inode_, rhs.inode_);
}

objbase::~objbase()
{}

void objbase::dispose() const {
	delete this;
}

const bs_inode* objbase::inode() const {
	return inode_;
}

type_descriptor objbase::bs_type() {
	return type_descriptor(
		BS_GET_TI(objbase), NULL, NULL, NULL, "blue_sky::objbase",
		"Base class of all BlueSky types"
	);
}

int objbase::bs_register_this() const {
	//return BS_KERNEL.register_instance(this);
	return 0;
}

int objbase::bs_free_this() const {
	//return BS_KERNEL.free_instance(this);
	return 0;
}

} /* namespace blue_sky */

// -----------------------------------------------------
// End of implementation class: object_base
// -----------------------------------------------------
