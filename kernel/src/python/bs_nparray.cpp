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

#include "py_array_converter.h"
#include "py_bs_exports.h"
#include "py_smart_ptr.h"

using namespace std;

namespace blue_sky {
// usefull typedefs
typedef bs_array< int, bs_nparray > bs_nparray_i;
typedef bs_array< long, bs_nparray > bs_nparray_l;
typedef bs_array< float, bs_nparray > bs_nparray_f;
typedef bs_array< double, bs_nparray > bs_nparray_d;

// bs_array< T, bs_nparray > instantiations
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (int, bs_nparray));
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (unsigned int, bs_nparray));
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (long, bs_nparray));
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (unsigned long, bs_nparray));
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (float, bs_nparray));
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (double, bs_nparray));

kernel::types_enum register_nparray() {
	kernel::types_enum te;
	te.push_back(bs_nparray_i::bs_type());
	te.push_back(bs_nparray_l::bs_type());
	te.push_back(bs_nparray_f::bs_type());
	te.push_back(bs_nparray_d::bs_type());
	return te;
}

namespace python {

// bs_nparray test function
template< class inp_array_t, class ret_array_t >
smart_ptr< ret_array_t > test_nparray(smart_ptr< inp_array_t > a, smart_ptr< inp_array_t > b) {
	// a little test on nparray resize
	smart_ptr< bs_nparray_d > tmp = BS_KERNEL.create_object(bs_nparray_d::bs_type());
	tmp->resize(10);
	tmp->resize(11);

	ulong sz = (ulong) std::min(a->size(), b->size());
	smart_ptr< ret_array_t > res = BS_KERNEL.create_object(ret_array_t::bs_type());
	*res = *a;
	res->assign(*a);
	res->assign(a->begin(), a->end());
	//*res = *a->clone();
	res->resize(sz + 1);
	for(ulong i = 0; i < sz; ++i)
		(*res)[i] = (*a)[i] + (*b)[i];
	(*res)[res->size() - 1] = 12345;
	//res->insert(1.);
	return res;
}

void py_export_nparray() {
	// export converters
	array_converters< int >           :: make_known();
	array_converters< unsigned int >  :: make_known();
	array_converters< long >          :: make_known();
	array_converters< unsigned long > :: make_known();
	array_converters< float >         :: make_known();
	array_converters< double >        :: make_known();

	// export test functions
	def("test_nparray_i", &test_nparray< bs_nparray_i, bs_array< int > >);
	def("test_nparray_l", &test_nparray< bs_nparray_i, bs_array< long > >);
	def("test_nparray_f", &test_nparray< bs_nparray_f, bs_array< float > >);
	def("test_nparray_d", &test_nparray< bs_array< double, bs_npvec_shared >,
		bs_array< double, bs_npvec_shared > >);
	//def("test_nparray_d", &test_nparray< bs_array< double, vector_traits >, bs_array< double, vector_traits > >);
	//def("test_nparray_d", &test_nparray< bs_nparray< double >, bs_array< double, vector_traits > >);
	//def("test_nparray_d", &test_nparray< double, bs_arrbase >);
}

}}

