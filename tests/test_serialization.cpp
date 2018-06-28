/// @file
/// @author uentity
/// @date 05.06.2018
/// @brief Serialization subsystem test
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#define BOOST_TEST_DYN_LINK
#include <bs/kernel.h>
#include <bs/log.h>

#include <bs/serialize/base_types.h>
#include <bs/serialize/array.h>

#include <boost/test/unit_test.hpp>
#include <iostream>

template<typename T>
auto test_json(const T& obj) {
	using namespace blue_sky;
	using namespace blue_sky::log;

	std::string dump;
	// dump object into string
	std::stringstream ss;
	{
		cereal::JSONOutputArchive ja(ss);
		ja(obj);
		dump = ss.str();
		bsout() << I("JSON dump\n: {}", dump) << end;
	}
	// load object from dump
	T obj1;
	{
		//std::istringstream is(dump);
		cereal::JSONInputArchive ja(ss);
		ja(obj1);
	}
	BOOST_TEST(obj->id() == obj1->id());
	return obj1;
}

BOOST_AUTO_TEST_CASE(test_serialization) {
	using namespace blue_sky;
	using namespace blue_sky::log;

	sp_obj obj = std::make_shared<objbase>();
	test_json(obj);

	// array
	using int_array = bs_array<int>;
	std::shared_ptr<int_array> arr = BS_KERNEL.create_object(int_array::bs_type(), 20);
	BOOST_TEST(arr);
	std::cout << "array size = " << arr->size() << std::endl;
	for(ulong i = 0; i < arr->size(); ++i)
		arr->ss(i) = i;

	auto arr1 = test_json(arr);
	BOOST_TEST(arr->size() == arr1->size());
	BOOST_TEST(std::equal(arr->begin(), arr->end(), arr1->begin()));

	// shared array
	using sint_array = bs_array<double, bs_vector_shared>;
	std::shared_ptr<sint_array> sarr = BS_KERNEL.create_object(sint_array::bs_type(), 20);
	BOOST_TEST(arr);

	auto sarr1 = test_json(sarr);
	BOOST_TEST(sarr->size() == sarr1->size());
	BOOST_TEST(std::equal(sarr->begin(), sarr->end(), sarr1->begin()));
}

