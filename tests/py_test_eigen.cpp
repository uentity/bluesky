/// @file
/// @author uentity
/// @date 22.03.2019
/// @brief Misc Eigen arrays tests
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/python/common.h>
#include <bs/python/tensor.h>
#include <bs/serialize/tensor.h>
#include "test_serialization.h"

#include <bs/log.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include <iostream>

using namespace Eigen;
namespace py = pybind11;
using namespace py::literals;
using namespace blue_sky;
using blue_sky::log::end;

template<typename T, bool is_rowmajor = false>
auto gen_tensor(
	Index nx = 2, Index ny = 2, Index nz = 2
) -> Tensor<T, 3, is_rowmajor ? RowMajor : ColMajor> {
	constexpr auto layout = is_rowmajor ? RowMajor : ColMajor;
	using tensor_t = Tensor<T, 3, layout>;
	using idx_t = typename tensor_t::Index;

	auto dims = std::array<Index, 3>{nx, ny, nz};
	auto t = tensor_t(dims);

	T v = 0;
	for(idx_t i = 0; i < t.size(); ++i)
		t.data()[i] = ++v;


	bsout() << "Tensor data = {" << end;
	for(idx_t i = 0; i < nx; ++i)
		for(idx_t j = 0; j < ny; ++j)
			for(idx_t k = 0; k < nz; ++k)
				bsout() << "[{}, {}, {}] = {}" << i << j << k << t(i, j, k) << end;
	bsout() << "}" << end;

	test_saveload(t);

	return t;

	//using vector_t = Array<T, Dynamic, 1>;
	//return vector_t(Map<vector_t>(t.data(), t.size()));
}

void test_eigen(py::module& m) {
	m.def("gen_tensor_df", &gen_tensor<double>, "nx"_a = 2, "ny"_a = 2, "nz"_a = 2);
	m.def("gen_tensor_if", &gen_tensor<int>, "nx"_a = 2, "ny"_a = 2, "nz"_a = 2);
	m.def("gen_tensor_dc", &gen_tensor<double, true>, "nx"_a = 2, "ny"_a = 2, "nz"_a = 2);
	m.def("gen_tensor_ic", &gen_tensor<int, true>, "nx"_a = 2, "ny"_a = 2, "nz"_a = 2);
}

