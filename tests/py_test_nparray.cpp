/// @file
/// @author uentity
/// @date 11.01.2017
/// @brief Test BS <-> numpy array interoperability
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/


#include <bs/bs.h>
#include <bs/python/nparray.h>

#include <iostream>
#include <algorithm>

using namespace blue_sky;
namespace py = pybind11;
using namespace py::literals;

using dvector = blue_sky::bs_numpy_array< double >;
using sp_dvector = std::shared_ptr< dvector >;
using ivector = blue_sky::bs_numpy_array< long >;
using sp_ivector = std::shared_ptr< ivector >;
using uivector = blue_sky::bs_numpy_array< unsigned long >;
using sp_uivector = std::shared_ptr< uivector >;

template< class Ostream, class T, template< class > class traits >
Ostream& operator <<(Ostream& os, const bs_array< T, traits >& A) {
	for(auto v : A) {
		os << v << ' ';
	}
	return os;
}

template<typename T>
void double_vec(const std::shared_ptr<blue_sky::bs_numpy_array<T>>& X, std::size_t extent = 2) {
	auto& x = *X;
	std::cout << __func__ << ": source: " << x << std::endl;
	try {
		x.resize(x.size() + extent, T(42));
	}
	catch(const std::exception& e) {
		std::cerr << __func__ << ": " << e.what() << std::endl;
	}
	std::transform(
		x.begin(), x.end(), x.begin(),
		[](double v){ return v*2; }
	);
	std::cout << __func__ << ": result: " << x << std::endl;
}

template<typename T>
auto gen_vec(std::size_t sz = 10, T fill_with = 42) -> std::shared_ptr< blue_sky::bs_numpy_array<T> > {
	using vec_t = blue_sky::bs_numpy_array<T>;
	auto res = std::make_shared< vec_t >();
	res->resize(sz);
	std::fill(res->begin(), res->end(), fill_with);
	return res;
}

// bs_nparray test function
template< class inp_array_t, class ret_array_t >
auto test_nparray_d(
	std::shared_ptr< inp_array_t > a, std::shared_ptr< inp_array_t > b
) -> std::shared_ptr< ret_array_t > {
	// a little test on nparray resize
	sp_dvector tmp = kernel::tfactory::create_object(dvector::bs_type());
	std::cout << "tmp created = " << tmp->size() << std::endl;
	tmp->resize(10);
	std::cout << "tmp resized = " << tmp->size() << std::endl;

	ulong sz = (ulong) std::min(a->size(), b->size());
	std::shared_ptr< ret_array_t > res = kernel::tfactory::create_object(ret_array_t::bs_type());
	res->assign(*a);
	res->assign(a->begin(), a->end());
	// res IS a (both reference same buffer)
	*res = *a;
	//*res = *a->clone();
	res->resize(sz + 1);
	using T = typename ret_array_t::value_type;
	T sum = 0;
	for(ulong i = 0; i < sz; ++i) {
		(*res)[i] = (*a)[i] + (*b)[i];
		sum += (*res)[i];
	}
	(*res)[res->size() - 1] = sum;
	return res;
}

void test_nparray(py::module& m) {
	m.def("double_vec", &double_vec<double>, "X"_a, "extent"_a = 2,
		"A function that doubles vector values");
	m.def("gen_vec", &gen_vec<double>, "sz"_a = 10, "fill_with"_a = 42);

	m.def("double_ivec", &double_vec<long>, "X"_a, "extent"_a = 2,
		"A function that doubles vector values");
	m.def("gen_ivec", &gen_vec<long>, "sz"_a = 10, "fill_with"_a = 42);

	m.def("double_uivec", &double_vec<unsigned long>, "X"_a, "extent"_a = 2,
		"A function that doubles vector values");
	m.def("gen_uivec", &gen_vec<unsigned long>, "sz"_a = 10, "fill_with"_a = 42);

	m.def("test_nparray_d", &test_nparray_d< dvector, dvector >);
}
