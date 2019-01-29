/// @file
/// @author uentity
/// @date 18.10.2016
/// @brief Test BS type factory
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#define BOOST_TEST_DYN_LINK
#include "test_objects.h"
#include <bs/type_macro.h>
#include <bs/kernel/types_factory.h>
#include <bs/log.h>

#include <boost/test/unit_test.hpp>
#include <iostream>

namespace blue_sky {

///////////////////////////////////////////////////////////////////////////////
//  Declarations of classes below are in `test_objects.h`
//
/*-----------------------------------------------------------------------------
 * type_descriptor for `person`
 *-----------------------------------------------------------------------------*/
// define free function that returns bs_person singleton
sp_obj create_single_person() {
	static bs_person P("[SINGLE]");
	auto noop_del = [](const bs_person*) {};
	return std::shared_ptr< bs_person >(&P, noop_del);
}

BS_TYPE_IMPL(bs_person, objbase, "bs_person", "BS Person", false, false)
BS_TYPE_ADD_CONSTRUCTOR(bs_person, (const char*))
BS_TYPE_ADD_CONSTRUCTOR(bs_person, (double))
BS_TYPE_ADD_CONSTRUCTOR(bs_person, (const char*, double))
// add free function constructor
BS_TYPE_ADD_DEF_CONSTRUCTOR(bs_person)
//BS_TYPE_ADD_CONSTRUCTOR_F(bs_person, create_single_person)

// these two should coincide
BS_TYPE_ADD_CONSTRUCTOR(bs_person, (const std::string&))
BS_TYPE_ADD_CONSTRUCTOR(bs_person, (std::string))

BS_TYPE_ADD_CONSTRUCTOR(bs_person, (const std::string&, double))
BS_TYPE_ADD_COPY_CONSTRUCTOR(bs_person)

BS_REGISTER_TYPE("unit_test", bs_person)

/*-----------------------------------------------------------------------------
 * .. for `my_strategy` templated class
 *-----------------------------------------------------------------------------*/
// register strategy
BS_TYPE_IMPL_INL_T1(my_strategy, int)
BS_TYPE_IMPL_INL_T1(my_strategy, double)
BS_TYPE_ADD_CONSTRUCTOR(my_strategy< int >, (const char*))
BS_TYPE_ADD_CONSTRUCTOR(my_strategy< double >, (const char*))
BS_TYPE_ADD_CONSTRUCTOR(my_strategy< int >, (const std::string&))
BS_TYPE_ADD_CONSTRUCTOR(my_strategy< double >, (const std::string&))

BS_REGISTER_TYPE("unit_test", my_strategy< int >)
BS_REGISTER_TYPE("unit_test", my_strategy< double >)

/*-----------------------------------------------------------------------------
 * .. for `my_strategy`
 *-----------------------------------------------------------------------------*/
// define free function that returns bs_person singleton
template< class T >
sp_obj create_single_uber(T val) {
	using uber_t = uber_type< T, my_strategy< T > >;
	static uber_t P("[SINGLE]");
	P.add_value(val);

	auto noop_del = [](const uber_t*) {};
	return std::shared_ptr< uber_t >(&P, noop_del);
}

BS_TYPE_IMPL_INL_T(uber_type, (int, my_strategy< int >))
BS_TYPE_IMPL_INL_T(uber_type, (double, my_strategy< double >))
BS_TYPE_ADD_CONSTRUCTOR_T(uber_type, (int, my_strategy< int >), (std::string))
BS_TYPE_ADD_CONSTRUCTOR_T(uber_type, (double, my_strategy< double >), (std::string))

// add free function constructors
BS_TYPE_ADD_CONSTRUCTOR_T_F(uber_type, (int, my_strategy< int >), create_single_uber, (int))
BS_TYPE_ADD_CONSTRUCTOR_T_F(uber_type, (double, my_strategy< double >), create_single_uber, (double))

BS_REGISTER_TYPE_T("unit_test", uber_type, (int, my_strategy< int >))
BS_REGISTER_TYPE_T("unit_test", uber_type, (double, my_strategy< double >))

} // eof blue_sky namespace

using namespace blue_sky;

BOOST_AUTO_TEST_CASE(test_type_descriptor) {
	std::cout << "\n\n*** testing bs_type_descriptor..." << std::endl;
	// register type first
	//BS_KERNEL.register_type(bs_person::bs_type());

	sp_person p = kernel::tfactory::create_object("bs_person");
	BOOST_TEST(p);
	if(p)
		std::cout << *p << std::endl;
	// fill name
	p = kernel::tfactory::create_object("bs_person", "John");
	BOOST_TEST(p);
	if(p)
		std::cout << *p << std::endl;
	// fill age with int - should fail
	p = kernel::tfactory::create_object("bs_person", 28);
	BOOST_TEST((p == nullptr));
	// fill age with double - should work
	p = kernel::tfactory::create_object("bs_person", 28.);
	BOOST_TEST(p);
	if(p)
		std::cout << *p << std::endl;
	// fill both age and name
	p = kernel::tfactory::create_object("bs_person", std::string("Sarah"), 33.);
	BOOST_TEST(p);
	if(p)
		std::cout << *p << std::endl;
	// make copy
	sp_person p1 = kernel::tfactory::clone_object(p);
	BOOST_TEST(p1);
	if(p1)
		std::cout << "Copy is: " << *p1 << std::endl;

	// create strategy
	sp_strat< int > si = kernel::tfactory::create_object(my_strategy< int >::bs_type(), "integer strategy");
	BOOST_TEST(si);
	if(si) {
		std::cout << *si << std::endl;
	}
	sp_strat< double > sd = kernel::tfactory::create_object("my_strategy double", "double strategy");
	BOOST_TEST(sd);
	if(sd) std::cout << *sd << std::endl;

	// create uber_type
	sp_uber< int > ui = kernel::tfactory::create_object(uber_type< int, my_strategy< int > >::bs_type());
	BOOST_TEST(ui);
	if(ui) std::cout << *ui << std::endl;

	sp_uber< double > ud = kernel::tfactory::create_object("uber_type double my_strategy< double >", "I'm double");
	ud->add_value(42.);
	BOOST_TEST(ud);
	if(ud) std::cout << *ud << std::endl;

	// test using free function
	ud = kernel::tfactory::create_object("uber_type double my_strategy< double >", 27.5);
	BOOST_TEST(ud);
	ud = kernel::tfactory::create_object("uber_type double my_strategy< double >", 42.);
	BOOST_TEST(ud);
	if(ud) std::cout << *ud << std::endl;
}

