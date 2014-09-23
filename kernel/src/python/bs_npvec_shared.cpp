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

#include "bs_npvec_shared.h"
#include "bs_array.h"
#include "py_bs_exports.h"
#include "py_smart_ptr.h"

using namespace std;

namespace blue_sky {
// usefull typedefs
typedef bs_array< int                , bs_npvec_shared > bs_npvec_shared_i;
typedef bs_array< unsigned int       , bs_npvec_shared > bs_npvec_shared_ui;
typedef bs_array< long               , bs_npvec_shared > bs_npvec_shared_l;
typedef bs_array< long long          , bs_npvec_shared > bs_npvec_shared_ll;
typedef bs_array< unsigned long      , bs_npvec_shared > bs_npvec_shared_ul;
typedef bs_array< unsigned long long , bs_npvec_shared > bs_npvec_shared_ull;
typedef bs_array< float              , bs_npvec_shared > bs_npvec_shared_f;
typedef bs_array< double             , bs_npvec_shared > bs_npvec_shared_d;
typedef bs_array< std::string        , bs_npvec_shared > bs_npvec_shared_s;
typedef bs_array< std::wstring       , bs_npvec_shared > bs_npvec_shared_ws;

// bs_array< T, bs_nparray > instantiations
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (int, bs_npvec_shared));
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (unsigned int, bs_npvec_shared));
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (long, bs_npvec_shared));
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (long long, bs_npvec_shared));
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (unsigned long, bs_npvec_shared));
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (unsigned long long, bs_npvec_shared));
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (float, bs_npvec_shared));
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (double, bs_npvec_shared));
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (std::string, bs_npvec_shared));
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (std::wstring, bs_npvec_shared));

// explicit instantiations needed by VS when linking with clients
// I think this is just to overcome compiler strange behaviour (bugs?)
template class detail::bs_npvec_impl< bs_array_shared< int > >;
template class detail::bs_npvec_impl< bs_array_shared< unsigned int > >;
template class detail::bs_npvec_impl< bs_array_shared< long > >;
template class detail::bs_npvec_impl< bs_array_shared< unsigned long > >;
template class detail::bs_npvec_impl< bs_array_shared< long long > >;
template class detail::bs_npvec_impl< bs_array_shared< unsigned long long > >;
template class detail::bs_npvec_impl< bs_array_shared< float > >;
template class detail::bs_npvec_impl< bs_array_shared< double > >;
template class detail::bs_npvec_impl< bs_array_shared< std::string > >;
template class detail::bs_npvec_impl< bs_array_shared< std::wstring > >;

kernel::types_enum register_npvec_shared() {
	kernel::types_enum te;
	te.push_back(bs_npvec_shared_i::bs_type());
	te.push_back(bs_npvec_shared_ui::bs_type());
	te.push_back(bs_npvec_shared_l::bs_type());
	te.push_back(bs_npvec_shared_ll::bs_type());
	te.push_back(bs_npvec_shared_ul::bs_type());
	te.push_back(bs_npvec_shared_ull::bs_type());
	te.push_back(bs_npvec_shared_f::bs_type());
	te.push_back(bs_npvec_shared_d::bs_type());
	te.push_back(bs_npvec_shared_s::bs_type());
	te.push_back(bs_npvec_shared_ws::bs_type());
	return te;
}

}


