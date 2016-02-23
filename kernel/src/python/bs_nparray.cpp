/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

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
BLUE_SKY_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (int, bs_nparray));
BLUE_SKY_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (unsigned int, bs_nparray));
BLUE_SKY_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (long, bs_nparray));
BLUE_SKY_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (long long, bs_nparray));
BLUE_SKY_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (unsigned long, bs_nparray));
BLUE_SKY_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (unsigned long long, bs_nparray));
BLUE_SKY_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (float, bs_nparray));
BLUE_SKY_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (double, bs_nparray));
BLUE_SKY_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (std::string, bs_nparray));
BLUE_SKY_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (std::wstring, bs_nparray));

kernel::types_enum register_nparray() {
	kernel::types_enum te;
	te.push_back(bs_nparray_i::bs_type());
	te.push_back(bs_nparray_l::bs_type());
	te.push_back(bs_nparray_f::bs_type());
	te.push_back(bs_nparray_d::bs_type());

	te.push_back(bs_array< unsigned int      , bs_nparray >::bs_type());
	te.push_back(bs_array< long long         , bs_nparray >::bs_type());
	te.push_back(bs_array< unsigned long     , bs_nparray >::bs_type());
	te.push_back(bs_array< unsigned long long, bs_nparray >::bs_type());
	te.push_back(bs_array< std::string       , bs_nparray >::bs_type());
	te.push_back(bs_array< std::wstring      , bs_nparray >::bs_type());
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
	array_converters< int >               :: make_known();
	array_converters< unsigned int >      :: make_known();
	array_converters< long >              :: make_known();
	array_converters< long long >         :: make_known();
	array_converters< unsigned long >     :: make_known();
	array_converters< unsigned long long> :: make_known();
	array_converters< float >             :: make_known();
	array_converters< double >            :: make_known();
	array_converters< std::string >       :: make_known();
	array_converters< std::wstring >      :: make_known();

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

