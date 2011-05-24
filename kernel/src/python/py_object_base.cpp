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
#ifdef BSPY_EXPORTING_PLUGIN
#include <boost/python.hpp>
#endif

#include "bs_object_base.h"
#include "bs_link.h"
#include "bs_messaging.h"
#include "py_bs_exports.h"
#include "py_smart_ptr.h"
//#include <boost/python/python.h>

namespace blue_sky {
// we need special comparison functions, because object are held as smart_ptr
inline bool operator==(const objbase& lhs, const objbase& rhs) {
	return (&lhs == &rhs);
}
inline bool operator!=(const objbase& lhs, const objbase& rhs) {
	return !(lhs == rhs);
}

namespace python {

using namespace boost::python;

namespace {
// objbase wrapper
class objbase_pyw : public objbase, public wrapper< objbase > {
public:
	type_descriptor bs_resolve_type() const {
		return this->get_override("bs_resolve_type")();
	}

	void dispose() const {
		if(override f = this->get_override("dispose"))
			f();
		else
			objbase::dispose();
	}

	void default_dispose() const {
		objbase::dispose();
	}
};

}

// exporting function
void py_bind_objbase() {
	// export bs_slot wrapper
	class_<
		objbase_pyw,
		smart_ptr< objbase_pyw, true >,
		bases< bs_messaging >,
		boost::noncopyable
		>
	("objbase", no_init)
		.def("bs_type", &objbase::bs_type)
		.def("bs_resolve_type", pure_virtual(&objbase::bs_resolve_type))
		.def("dispose", &objbase::dispose, &objbase_pyw::default_dispose)
		.def("bs_register_this", &objbase::bs_register_this)
		.def("bs_free_this", &objbase::bs_free_this)
		.def("inode", &objbase::inode, return_internal_reference<>())
		.staticmethod("bs_type")
		.def(self == self)
		.def(self != self)
		;
	// register smart_ptr conversions
	implicitly_convertible< smart_ptr< objbase_pyw, true >, blue_sky::smart_ptr< objbase, true > >();
	implicitly_convertible< smart_ptr< objbase, true >, blue_sky::smart_ptr< bs_messaging, true > >();
	register_ptr_to_python< blue_sky::smart_ptr< objbase, true > >();
}

}}	// eof namespace blue_sky::python

