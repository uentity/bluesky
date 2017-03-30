/// @file
/// @author uentity
/// @date 11.01.2017
/// @brief Test BS <-> numpy array interoperability
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/


#include <bs/kernel.h>
#include <bs/python/nparray.h>
#include <bs/compat/array.h>

#include <pybind11/pybind11.h>
#include <iostream>
#include <algorithm>

using namespace blue_sky;
namespace py = pybind11;

using dvector = blue_sky::bs_numpy_array< double >;
using sp_dvector = std::shared_ptr< dvector >;

template< class T >
struct test { T m_; };

using tint = test< int >;

//PYBIND11_DECLARE_HOLDER_TYPE(dvector, sp_dvector);

//namespace pybind11 { namespace detail {
//
//template<> struct type_caster< sp_dvector > {
//	typedef py::array_t< double > numpy_t;
//	PYBIND11_TYPE_CASTER(sp_dvector, _("dvector"));
//
//	bool load(handle src, bool) {
//		value = std::make_shared< dvector >(numpy_t::ensure(src));
//		return static_cast< bool >(value.get());
//	}
//
//	static handle cast(const sp_dvector& src, return_value_policy, handle) {
//		return src.get() ? src->get_container()->inc_ref() : handle();
//	}
//};
//
//}}

template< class Ostream, class T, template< class > class traits >
Ostream& operator <<(Ostream& os, const bs_array< T, traits >& A) {
	for(auto v : A) {
		os << v << ' ';
	}
	return os;
}

void double_vec(const sp_dvector& X) {
	dvector& x = *X;
	std::cout << "Source: " << x << std::endl;
	try {
		x.resize(x.size() + 2, 42.);
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

sp_dvector gen_vec() {
	//sp_dvector res = std::make_shared< dvector >(10);
	sp_dvector res = std::make_shared< dvector >();
	res->resize(10);
	std::fill(res->begin(), res->end(), 42.);
	return res;
}

// bs_nparray test function
template< class inp_array_t, class ret_array_t >
std::shared_ptr< ret_array_t > test_nparray(std::shared_ptr< inp_array_t > a, std::shared_ptr< inp_array_t > b) {
	// a little test on nparray resize
	sp_dvector tmp = BS_KERNEL.create_object(dvector::bs_type());
	std::cout << "tmp created = " << tmp->size() << std::endl;
	tmp->resize(10);
	std::cout << "tmp resized = " << tmp->size() << std::endl;

	ulong sz = (ulong) std::min(a->size(), b->size());
	std::shared_ptr< ret_array_t > res = BS_KERNEL.create_object(ret_array_t::bs_type());
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

PYBIND11_PLUGIN(test_nparray) {
	// bind all functions to test_pymod
	py::module m("test_nparray");
	//auto subm = m.def_submodule("test_nparray");
	//std::cout << m.ptr() << ' ' << subm.ptr() << std::endl;

	m.def("double_vec", &double_vec, "A function that doubles vector values");
	m.def("gen_vec", &gen_vec);

	m.def("test_nparray_d", &test_nparray< dvector, dvector >);

	return m.ptr();

}
