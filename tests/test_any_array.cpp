/// @file
/// @author uentity
/// @date 03.04.2017
/// @brief Test any_array
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#define BOOST_TEST_DYN_LINK
#include <bs/kernel.h>
#include <bs/error.h>
#include <bs/objbase.h>

#include <boost/test/unit_test.hpp>
#include <iostream>

NAMESPACE_BEGIN(blue_sky)

struct foo : public objbase { BS_TYPE_DECL };

BS_TYPE_IMPL(foo, objbase, "test_any_foo", "Test foo type", true, true)
BS_REGISTER_RT_TYPE(foo)

NAMESPACE_END(blue_sky)


BOOST_AUTO_TEST_CASE(test_any_array) {
	using namespace blue_sky;

	std::cout << "\n\n*** testing any_array..." << std::endl;

	// map of string key -> any value
	auto a = str_any_array();
	a["1"] = 42;
	a["2"] = error("OMG!");
	a["3"] = 84.7;
	a["4"] = "test";
	a["5"] = std::string("hello");
	// check overall size
	BOOST_TEST(a.size() == 5u);

	// check cout of different elements
	BOOST_TEST(a.size< int >() == 1u);
	// following should give same results
	BOOST_TEST(a.size< std::string >() == 1u);
	BOOST_TEST(a.size< const std::string& >() == 1u);

	BOOST_TEST(a.size< double >() == 1u);
	BOOST_TEST(a.size< error >() == 1u);
	BOOST_TEST(a.size< const char* >() == 1u);
	BOOST_TEST(a.size< char* >() == 0u);

	// check elements
	int t1 = a.ss("1");
	BOOST_TEST(t1 == 42);
	// conversion to double from int is not allowed
	double t2 = 0;
	a.at("1", t2);
	BOOST_TEST(t2 == 0);
	std::cout << "a[""4""] should be test, and really: " << a.ss< const char* >("4") << std::endl;
	std::cout << "a[""2""] is error: " << a.ss< error >("2") << std::endl;

	// map of index -> any value
	auto b = idx_any_array(3);
	b[0] = 42;
	b[1] = a["2"];
	b[2] = 84.7;
	b.push_back("test");
	b.push_back(std::string("hello"));
	// check overall size
	BOOST_TEST(b.size() == 5u);

	// following should give same results
	BOOST_TEST(b.size< std::string >() == 1u);
	BOOST_TEST(b.size< const std::string& >() == 1u);

	BOOST_TEST(b.size< double >() == 1u);
	BOOST_TEST(b.size< error >() == 1u);
	BOOST_TEST(b.size< const char* >() == 1u);
	BOOST_TEST(b.size< char* >() == 0u);

	// test assignment via subscript
	b.ss< std::string >(4) = "Hello World!";
	BOOST_TEST(b.ss< std::string >(4) == "Hello World!");
	std::cout << b.ss< std::string >(4) << std::endl;

	// check at()
	// should fail to default value
	int t3 = b.at(3, 33);
	BOOST_CHECK(t3 == 33);
	// should succeed
	t3 = b.at(0, 33);
	BOOST_CHECK(t3 == 42);

	// check extract()
	// shouldn't change value
	t3 = 33;
	b.extract(4, t3);
	BOOST_CHECK(t3 == 33);
	b.extract(8, t3);
	BOOST_CHECK(t3 == 33);
	// should succeed
	b.extract(0, t3);
	BOOST_CHECK(t3 == 42);

	// check kernel tables
	auto& kfoo_tbl = BS_KERNEL.pert_idx_any_array("test_any_foo");
	kfoo_tbl.resize(4);
	kfoo_tbl[0] = "1";
	kfoo_tbl[1] = 2;
	kfoo_tbl[2] = "Hello World!";
	kfoo_tbl[3] = std::string("This is string");

	auto& kfoo_tbl1 = BS_KERNEL.pert_idx_any_array(foo::bs_type());
	BOOST_CHECK(&kfoo_tbl == &kfoo_tbl1);

	BOOST_CHECK(kfoo_tbl1.values< int >().size() == 1u);
	BOOST_CHECK(kfoo_tbl1.values< std::string >().size() == 1u);
	BOOST_CHECK(kfoo_tbl1.values< const char* >().size() == 2u);
	BOOST_CHECK(kfoo_tbl1.values< double >().size() == 0u);

	// print keys
	std::cout << "test_any_foo array keys:";
	for(auto& k : kfoo_tbl.keys())
		std::cout << ' ' << k;
	std::cout << std::endl;

	// save state
	auto str_state = kfoo_tbl.values_map< std::string >();
	std::cout << "test_any_foo array std::string values:";
	for(auto& k : str_state)
		std::cout << " [" << k.first << ": " << k.second << "]";
	std::cout << std::endl;

	// modify state
	kfoo_tbl.ss< std::string >(3) = "NO!";
	BOOST_TEST(kfoo_tbl.ss< std::string >(3) == "NO!");
	// restore state
	kfoo_tbl = str_state;
	BOOST_TEST(kfoo_tbl.ss< std::string >(3) == str_state[3]);
}

