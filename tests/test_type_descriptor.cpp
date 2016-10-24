/// @file
/// @author uentity
/// @date 18.10.2016
/// @brief Test BS type factory
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#define BOOST_TEST_DYN_LINK
#include <bs/objbase.h>
#include <bs/type_macro.h>
#include <bs/kernel.h>
#include <bs/log.h>
#include <boost/test/unit_test.hpp>
#include <iostream>

namespace blue_sky {

class bs_person : public objbase {
public:
	bs_person() : name_("[NONAME]"), age_(0) {}

	bs_person(const char* name) : name_(name), age_(0) {}

	bs_person(double age) : name_("[NONAME]"), age_(age) {}

	bs_person(const std::string& name, double age = 0) : name_(name), age_(age) {}

	template< class Ostream >
	friend Ostream& operator<<(Ostream& os, const bs_person& p) {
		return os << "Person name = " << p.name_ << ", age = " << p.age_;
	}

	std::string name_;
	double age_;

	BS_TYPE_DECL
};
using sp_person = std::shared_ptr< bs_person >;

BS_TYPE_IMPL(bs_person, objbase, "bs_person", "BS Person", false, false)
BS_TYPE_ADD_EMPTY_CONSTRUCTOR(bs_person)
BS_TYPE_ADD_CONSTRUCTOR(bs_person, 1, (const char*))
BS_TYPE_ADD_CONSTRUCTOR(bs_person, 1, (double))
BS_TYPE_ADD_CONSTRUCTOR(bs_person, 1, (const std::string&))
BS_TYPE_ADD_CONSTRUCTOR(bs_person, 2, (const std::string&, double))
BS_TYPE_ADD_COPY_CONSTRUCTOR(bs_person)

}

using namespace blue_sky;

BOOST_AUTO_TEST_CASE(test_type_descriptor) {
	// register type first
	BS_KERNEL.register_type(bs_person::bs_type());

	sp_person p = BS_KERNEL.create_object("bs_person");
	BOOST_TEST(p);
	if(p)
		std::cout << *p << std::endl;
	// fill name
	p = BS_KERNEL.create_object("bs_person", "John");
	BOOST_TEST(p);
	if(p)
		std::cout << *p << std::endl;
	// fill age with int - should fail
	p = BS_KERNEL.create_object("bs_person", 28);
	BOOST_TEST((p == nullptr));
	// fill age with double - should work
	p = BS_KERNEL.create_object("bs_person", 28.);
	BOOST_TEST(p);
	if(p)
		std::cout << *p << std::endl;
	// fill both age and name
	p = BS_KERNEL.create_object("bs_person", std::string("Sarah"), 33.);
	BOOST_TEST(p);
	if(p)
		std::cout << *p << std::endl;
	// make copy
	sp_person p1 = BS_KERNEL.create_object_copy(p);
	BOOST_TEST(p1);
	if(p1)
		std::cout << "copy is: " << *p1 << std::endl;
}

