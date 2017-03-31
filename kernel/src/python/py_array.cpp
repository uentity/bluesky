/// @file
/// @author uentity
/// @date 11.01.2017
/// @brief bs_nupy_array specialization and registering
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/kernel.h>
#include <bs/python/nparray.h>

using namespace std;

namespace blue_sky {

BS_TYPE_IMPL_INL_T1(bs_numpy_array, int);
BS_TYPE_IMPL_INL_T1(bs_numpy_array, unsigned int);
BS_TYPE_IMPL_INL_T1(bs_numpy_array, long);
BS_TYPE_IMPL_INL_T1(bs_numpy_array, long long);
BS_TYPE_IMPL_INL_T1(bs_numpy_array, unsigned long);
BS_TYPE_IMPL_INL_T1(bs_numpy_array, unsigned long long);
BS_TYPE_IMPL_INL_T1(bs_numpy_array, float);
BS_TYPE_IMPL_INL_T1(bs_numpy_array, double);

BS_REGISTER_TYPE("kernel", bs_numpy_array<int>);
BS_REGISTER_TYPE("kernel", bs_numpy_array<unsigned int>);
BS_REGISTER_TYPE("kernel", bs_numpy_array<long>);
BS_REGISTER_TYPE("kernel", bs_numpy_array<long long>);
BS_REGISTER_TYPE("kernel", bs_numpy_array<unsigned long>);
BS_REGISTER_TYPE("kernel", bs_numpy_array<unsigned long long>);
BS_REGISTER_TYPE("kernel", bs_numpy_array<float>);
BS_REGISTER_TYPE("kernel", bs_numpy_array<double>);

} /* namespace blue_sky */

