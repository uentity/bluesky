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
void double_vec(const std::shared_ptr<blue_sky::bs_numpy_array<T>>& X) {
	auto& x = *X;
	std::cout << "Source: " << x << std::endl;
	try {
		x.resize(x.size() + 2, T(42));
	}
	catch(const std::exception& e) {
		std::cout << e.what() << std::endl;
	}
	std::transform(
		x.begin(), x.end(), x.begin(),
		[](double v){ return v*2; }
	);
	std::cout << "Result: " << x << std::endl;
}

template<typename T>
auto gen_vec() -> std::shared_ptr< blue_sky::bs_numpy_array<T> > {
	using vec_t = blue_sky::bs_numpy_array<T>;
	auto res = std::make_shared< vec_t >();
	res->resize(10);
	std::fill(res->begin(), res->end(), T(42));
	return res;
}

// bs_nparray test function
template< class inp_array_t, class ret_array_t >
std::shared_ptr< ret_array_t > test_nparray(std::shared_ptr< inp_array_t > a, std::shared_ptr< inp_array_t > b) {
	// a little test on nparray resize
	sp_dvector tmp = kernel::tfactory::create_object(dvector::bs_type());
	std::cout << "tmp created = " << tmp->size() << std::endl;
	tmp->resize(10);
	std::cout << "tmp resized = " << tmp->size() << std::endl;

	ulong sz = (ulong) std::min(a->size(), b->size());
	std::shared_ptr< ret_array_t > res = kernel::tfactory::create_object(ret_array_t::bs_type());
	*res = *a;
	res->assign(*a);
	res->assign(a->begin(), a->end());
	//*res = *a->clone();
	res->resize(sz + 1);
	for(ulong i = 0; i < sz; ++i)
		(*res)[i] = (*a)[i] + (*b)[i];
	(*res)[res->size() - 1] = 12345;
	return res;
}

PYBIND11_MODULE(test_nparray, m) {
	m.def("double_vec", &double_vec<double>, "A function that doubles vector values");
	m.def("gen_vec", &gen_vec<double>);

	m.def("double_ivec", &double_vec<long>, "A function that doubles vector values");
	m.def("gen_ivec", &gen_vec<long>);

	m.def("double_uivec", &double_vec<unsigned long>, "A function that doubles vector values");
	m.def("gen_uivec", &gen_vec<unsigned long>);

	m.def("test_nparray_d", &test_nparray< dvector, dvector >);
}
