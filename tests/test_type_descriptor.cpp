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

BLUE_SKY_PLUGIN_DESCRIPTOR("test_type_descriptor", "1.0", "Types factory unit test");

namespace blue_sky {

/*-----------------------------------------------------------------------------
 * test simple class
 *-----------------------------------------------------------------------------*/
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
BS_TYPE_ADD_DEF_CONSTRUCTOR(bs_person)
BS_TYPE_ADD_CONSTRUCTOR(bs_person, (const char*))
BS_TYPE_ADD_CONSTRUCTOR(bs_person, (double))

// these two should coincide
BS_TYPE_ADD_CONSTRUCTOR(bs_person, (const std::string&))
BS_TYPE_ADD_CONSTRUCTOR(bs_person, (std::string))

BS_TYPE_ADD_CONSTRUCTOR(bs_person, (const std::string&, double))
BS_TYPE_ADD_COPY_CONSTRUCTOR(bs_person)

BS_REGISTER_TYPE(bs_person)

/*-----------------------------------------------------------------------------
 * test templated class
 *-----------------------------------------------------------------------------*/
template< class T >
struct my_strategy : public objbase {
	using cont_type = std::vector< T >;

	my_strategy() : name_("[NONAME]") {}
	my_strategy(const my_strategy&) = default;
	my_strategy(my_strategy&&) = default;
	my_strategy(const std::string& name) : name_(name) {}

	std::string name_;

	//template< class Ostream >
	friend std::ostream& operator<<(std::ostream& os, const my_strategy& s) {
		return os << "Strategy type : " << bs_type().type_name() << "; strategy name = " << s.name_;
	}

	// expect that only one strategy can exist
	BS_TYPE_DECL_INL(my_strategy, objbase, "my_strategy", "Strategy", true, true)
};
template< class T >
using sp_strat = std::shared_ptr< my_strategy< T > >;

// register strategy
BS_TYPE_IMPL_INL_T1(my_strategy, int)
BS_TYPE_IMPL_INL_T1(my_strategy, double)
BS_TYPE_ADD_CONSTRUCTOR(my_strategy< int >, (const char*))
BS_TYPE_ADD_CONSTRUCTOR(my_strategy< double >, (const char*))
BS_TYPE_ADD_CONSTRUCTOR(my_strategy< int >, (const std::string&))
BS_TYPE_ADD_CONSTRUCTOR(my_strategy< double >, (const std::string&))

BS_REGISTER_TYPE(my_strategy< int >)
BS_REGISTER_TYPE(my_strategy< double >)

/*-----------------------------------------------------------------------------
 * test complex templated class
 *-----------------------------------------------------------------------------*/
// Strategy is passed with tuple argument just for testing
template< class T, class Strategy >
class uber_type : public objbase {
public:
	using uber_T = T;
	using cont_T = typename Strategy::cont_type;

	uber_type() : name_("[NONAME]") {};
	uber_type(const uber_type&) = default;
	uber_type(const char* name) : name_(name) {}
	uber_type(const std::string& name) : name_(name) {}

	void add_value(uber_T val) { storage_.emplace_back(std::move(val)); }

	//template< class Ostream >
	friend std::ostream& operator<<(std::ostream& os, const uber_type& s) {
		os << "Uber type: " << bs_type().type_name() << "; Uber name = " << s.name_ <<
			" has elements [" << s.storage_.size() << "]: ";
		for(auto& v : s.storage_) {
			os << v;
		}
		return os;
	}

	T value_;
	cont_T storage_;
	std::string name_;

	BS_TYPE_DECL_INL_BEGIN(uber_type, objbase, "uber", "Uber complex type", true, false)
		td.add_constructor< uber_type, const char* >();
		td.add_copy_constructor< uber_type >();
	BS_TYPE_DECL_INL_END
};
template< class T >
using sp_uber = std::shared_ptr< uber_type< T, my_strategy< T > > >;

BS_TYPE_IMPL_INL_T(uber_type, (int, my_strategy< int >))
BS_TYPE_IMPL_INL_T(uber_type, (double, my_strategy< double >))
BS_TYPE_ADD_CONSTRUCTOR_T(uber_type, (int, my_strategy< int >), (std::string))
BS_TYPE_ADD_CONSTRUCTOR_T(uber_type, (double, my_strategy< double >), (std::string))

BS_REGISTER_TYPE_T(uber_type, (int, my_strategy< int >))
BS_REGISTER_TYPE_T(uber_type, (double, my_strategy< double >))

} // eof blue_sky namespace

using namespace blue_sky;

BOOST_AUTO_TEST_CASE(test_type_descriptor) {
	std::cout << "*** testing bs_type_descriptor..." << std::endl;
	// register type first
	//BS_KERNEL.register_type(bs_person::bs_type());

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
		std::cout << "Copy is: " << *p1 << std::endl;

	// create strategy
	sp_strat< int > si = BS_KERNEL.create_object(my_strategy< int >::bs_type(), "integer strategy");
	BOOST_TEST(si);
	if(si) {
		std::cout << *si << std::endl;
	}
	sp_strat< double > sd = BS_KERNEL.create_object("my_strategy double", "double strategy");
	BOOST_TEST(sd);
	if(sd) std::cout << *sd << std::endl;

	// create uber_type
	sp_uber< int > ui = BS_KERNEL.create_object(uber_type< int, my_strategy< int > >::bs_type());
	BOOST_TEST(ui);
	if(ui) std::cout << *ui << std::endl;

	sp_uber< double > ud = BS_KERNEL.create_object("uber double my_strategy< double >", "I'm double");
	ud->add_value(42.);
	BOOST_TEST(ud);
	if(ud) std::cout << *ud << std::endl;
	//bsout() << *ud << bs_end;
}

