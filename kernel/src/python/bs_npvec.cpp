/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "bs_npvec.h"
#include "bs_array.h"
#include "py_bs_exports.h"
#include "py_smart_ptr.h"

// NOTE: explicit instantiations of bs_npvec_impl needed for VS
// I think this is just to overcome compiler strange behaviour (bugs?)
#define NPVEC_IMPL(T)                                    \
BLUE_SKY_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (T, bs_npvec)) \
template class detail::bs_npvec_impl< bs_vecbase_impl< T, std::vector< T > > >;

using namespace std;

namespace blue_sky {
// usefull typedefs
typedef bs_array< int                , bs_npvec > bs_npvec_i;
typedef bs_array< unsigned int       , bs_npvec > bs_npvec_ui;
typedef bs_array< long               , bs_npvec > bs_npvec_l;
typedef bs_array< long long          , bs_npvec > bs_npvec_ll;
typedef bs_array< unsigned long      , bs_npvec > bs_npvec_ul;
typedef bs_array< unsigned long long , bs_npvec > bs_npvec_ull;
typedef bs_array< float              , bs_npvec > bs_npvec_f;
typedef bs_array< double             , bs_npvec > bs_npvec_d;
typedef bs_array< std::string        , bs_npvec > bs_npvec_s;
typedef bs_array< std::wstring       , bs_npvec > bs_npvec_ws;

// instantiations
NPVEC_IMPL(int);
NPVEC_IMPL(unsigned int);
NPVEC_IMPL(long);
NPVEC_IMPL(unsigned long);
NPVEC_IMPL(long long);
NPVEC_IMPL(unsigned long long);
NPVEC_IMPL(float);
NPVEC_IMPL(double);
NPVEC_IMPL(std::string);
NPVEC_IMPL(std::wstring);

kernel::types_enum register_npvec() {
	kernel::types_enum te;
	te.push_back(bs_npvec_i::bs_type());
	te.push_back(bs_npvec_ui::bs_type());
	te.push_back(bs_npvec_l::bs_type());
	te.push_back(bs_npvec_ll::bs_type());
	te.push_back(bs_npvec_ul::bs_type());
	te.push_back(bs_npvec_ull::bs_type());
	te.push_back(bs_npvec_f::bs_type());
	te.push_back(bs_npvec_d::bs_type());
	te.push_back(bs_npvec_s::bs_type());
	te.push_back(bs_npvec_ws::bs_type());
	return te;
}

}

