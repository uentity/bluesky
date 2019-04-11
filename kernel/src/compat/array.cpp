/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief bs_array specializations and registering
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/kernel/types_factory.h>
#include <bs/compat/array.h>
#include <bs/compat/array_eigen_traits.h>

using namespace std;

namespace blue_sky {

BS_TYPE_IMPL_INL_T(bs_array, (int, vector_traits));
BS_TYPE_IMPL_INL_T(bs_array, (unsigned int, vector_traits));
BS_TYPE_IMPL_INL_T(bs_array, (intmax_t, vector_traits));
BS_TYPE_IMPL_INL_T(bs_array, (uintmax_t, vector_traits));
BS_TYPE_IMPL_INL_T(bs_array, (float, vector_traits));
BS_TYPE_IMPL_INL_T(bs_array, (double, vector_traits));

BS_REGISTER_TYPE_T("kernel", bs_array, (int, vector_traits));
BS_REGISTER_TYPE_T("kernel", bs_array, (unsigned int, vector_traits));
BS_REGISTER_TYPE_T("kernel", bs_array, (intmax_t, vector_traits));
BS_REGISTER_TYPE_T("kernel", bs_array, (uintmax_t, vector_traits));
BS_REGISTER_TYPE_T("kernel", bs_array, (float, vector_traits));
BS_REGISTER_TYPE_T("kernel", bs_array, (double, vector_traits));

BS_TYPE_IMPL_INL_T(bs_array, (int, bs_vector_shared));
BS_TYPE_IMPL_INL_T(bs_array, (unsigned int, bs_vector_shared));
BS_TYPE_IMPL_INL_T(bs_array, (intmax_t, bs_vector_shared));
BS_TYPE_IMPL_INL_T(bs_array, (uintmax_t, bs_vector_shared));
BS_TYPE_IMPL_INL_T(bs_array, (float, bs_vector_shared));
BS_TYPE_IMPL_INL_T(bs_array, (double, bs_vector_shared));

BS_REGISTER_TYPE_T("kernel", bs_array, (int, bs_vector_shared));
BS_REGISTER_TYPE_T("kernel", bs_array, (unsigned int, bs_vector_shared));
BS_REGISTER_TYPE_T("kernel", bs_array, (intmax_t, bs_vector_shared));
BS_REGISTER_TYPE_T("kernel", bs_array, (uintmax_t, bs_vector_shared));
BS_REGISTER_TYPE_T("kernel", bs_array, (float, bs_vector_shared));
BS_REGISTER_TYPE_T("kernel", bs_array, (double, bs_vector_shared));

}	// end of blue_sky namespace

