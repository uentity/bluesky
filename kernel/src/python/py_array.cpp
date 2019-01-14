/// @file
/// @author uentity
/// @date 11.01.2017
/// @brief bs_nupy_array specialization and registering
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/kernel/types_factory.h>
#include <bs/python/nparray.h>

#define INSTANTIATE_NUMPY_ARRAY(T)                                                \
BS_TYPE_IMPL_INL_T1(bs_numpy_array, T)                                            \
BS_TYPE_ADD_CONSTRUCTOR(bs_numpy_array< T >, (const std::vector< std::size_t >&)) \
BS_REGISTER_TYPE("kernel", bs_numpy_array<T>)

namespace blue_sky {

INSTANTIATE_NUMPY_ARRAY(int);
INSTANTIATE_NUMPY_ARRAY(unsigned int);
INSTANTIATE_NUMPY_ARRAY(long);
INSTANTIATE_NUMPY_ARRAY(long long);
INSTANTIATE_NUMPY_ARRAY(unsigned long);
INSTANTIATE_NUMPY_ARRAY(unsigned long long);
INSTANTIATE_NUMPY_ARRAY(float);
INSTANTIATE_NUMPY_ARRAY(double);

//BS_TYPE_IMPL_INL_T1(bs_numpy_array, int);
//BS_TYPE_IMPL_INL_T1(bs_numpy_array, unsigned int);
//BS_TYPE_IMPL_INL_T1(bs_numpy_array, long);
//BS_TYPE_IMPL_INL_T1(bs_numpy_array, long long);
//BS_TYPE_IMPL_INL_T1(bs_numpy_array, unsigned long);
//BS_TYPE_IMPL_INL_T1(bs_numpy_array, unsigned long long);
//BS_TYPE_IMPL_INL_T1(bs_numpy_array, float);
//BS_TYPE_IMPL_INL_T1(bs_numpy_array, double);
//
//BS_REGISTER_TYPE("kernel", bs_numpy_array<int>);
//BS_REGISTER_TYPE("kernel", bs_numpy_array<unsigned int>);
//BS_REGISTER_TYPE("kernel", bs_numpy_array<long>);
//BS_REGISTER_TYPE("kernel", bs_numpy_array<long long>);
//BS_REGISTER_TYPE("kernel", bs_numpy_array<unsigned long>);
//BS_REGISTER_TYPE("kernel", bs_numpy_array<unsigned long long>);
//BS_REGISTER_TYPE("kernel", bs_numpy_array<float>);
//BS_REGISTER_TYPE("kernel", bs_numpy_array<double>);

} /* namespace blue_sky */

