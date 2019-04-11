/// @file
/// @author uentity
/// @date 05.06.2018
/// @brief Serialization subsystem test
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#define BOOST_TEST_DYN_LINK
#include "test_objects.h"
#include "test_serialization.h"
#include <bs/kernel/kernel.h>
#include <bs/property.h>
#include <bs/propdict.h>

#include <bs/serialize/base_types.h>
#include <bs/serialize/array.h>
#include <cereal/types/vector.hpp>
#include <cereal/types/variant.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/chrono.hpp>

#include <boost/test/unit_test.hpp>
#include <iostream>

/*-----------------------------------------------------------------------------
 *  serialization impl for test classes
 *-----------------------------------------------------------------------------*/
using namespace blue_sky;

BSS_FCN_BEGIN(serialize, bs_person)
	ar(
		cereal::make_nvp("objbase", cereal::base_class<objbase>(&t)),
		cereal::make_nvp("name", t.name_),
		cereal::make_nvp("age", t.age_)
	);
BSS_FCN_END

BSS_FCN_EXPORT(serialize, bs_person)

BSS_FCN_BEGIN_T(serialize, my_strategy, 1)
	ar(
		cereal::make_nvp("objbase", cereal::base_class<objbase>(&t)),
		cereal::make_nvp("name", t.name_)
	);
BSS_FCN_END

BSS_REGISTER_TYPE_T(my_strategy, int)
BSS_REGISTER_TYPE_T(my_strategy, double)
BSS_FCN_EXPORT_T(serialize, my_strategy, int)
BSS_FCN_EXPORT_T(serialize, my_strategy, double)

BSS_FCN_BEGIN_T(serialize, uber_type, 2)
	ar(
		cereal::make_nvp("objbase", cereal::base_class<objbase>(&t)),
		cereal::make_nvp("value", t.value_),
		cereal::make_nvp("storage", t.storage_),
		cereal::make_nvp("name", t.name_)
	);
BSS_FCN_END

BSS_REGISTER_TYPE_EXT(uber_type, (int, my_strategy< int >))
BSS_REGISTER_TYPE_EXT(uber_type, (double, my_strategy< double >))
BSS_FCN_EXPORT_EXT(serialize, uber_type, (int, my_strategy< int >))
BSS_FCN_EXPORT_EXT(serialize, uber_type, (double, my_strategy< double >))

// register sample types

BSS_REGISTER_DYNAMIC_INIT(test_objects)

/*-----------------------------------------------------------------------------
 *  test unit
 *-----------------------------------------------------------------------------*/
using namespace blue_sky;

BOOST_AUTO_TEST_CASE(test_serialization) {
	using namespace blue_sky::log;

	std::cout << "\n\n*** testing serialization..." << std::endl;

	sp_obj obj = std::make_shared<objbase>();
	test_json(obj);

	// person
	sp_person P = kernel::tfactory::create_object(bs_person::bs_type(), std::string("Monkey"), double(22));
	BOOST_TEST(P);
	auto P1 = test_json(P);
	BOOST_TEST(P->name_ == P1->name_);

	// test NaN
	P = kernel::tfactory::create_object(
		bs_person::bs_type(), std::string("NaN"), std::numeric_limits<double>::quiet_NaN()
	);
	BOOST_TEST(P);
	P1 = test_json(P);
	BOOST_TEST(P1->age_ != P->age_);
	P = kernel::tfactory::create_object(
		bs_person::bs_type(), std::string("SNaN"), std::numeric_limits<double>::signaling_NaN()
	);
	BOOST_TEST(P);
	P1 = test_json(P);
	BOOST_TEST(P1->age_ != P->age_);

	// array
	using int_array = bs_array<int>;
	std::shared_ptr<int_array> arr = kernel::tfactory::create_object(int_array::bs_type(), 20);
	BOOST_TEST(arr);
	std::cout << "array size = " << arr->size() << std::endl;
	for(ulong i = 0; i < arr->size(); ++i)
		arr->ss(i) = i;

	auto arr1 = test_json(arr);
	BOOST_TEST(arr->size() == arr1->size());
	BOOST_TEST(std::equal(arr->begin(), arr->end(), arr1->begin()));

	// shared array
	using sd_array = bs_array<double, bs_vector_shared>;
	std::shared_ptr<sd_array> sarr = kernel::tfactory::create_object(sd_array::bs_type(), 20);
	BOOST_TEST(arr);
	std::cout << "array size = " << arr->size() << std::endl;
	for(ulong i = 0; i < arr->size(); ++i)
		arr->ss(i) = i;
	// test some corner values
	using dlimits = std::numeric_limits<double>;
	sarr->ss(0) = dlimits::min();
	sarr->ss(1) = dlimits::max();
	sarr->ss(2) = dlimits::lowest();
	sarr->ss(3) = dlimits::epsilon();
	sarr->ss(4) = dlimits::round_error();
	sarr->ss(5) = dlimits::infinity();
	sarr->ss(6) = dlimits::denorm_min();

	auto sarr1 = test_json(sarr);
	BOOST_TEST(sarr->size() == sarr1->size());
	BOOST_TEST(std::equal(sarr->begin(), sarr->end(), sarr1->begin()));

	// test properties serialization
	using namespace blue_sky::prop;
	propdict D = {{"A", "test2"}, {"B", 2L}, {"C", 42.}, {"D", std::vector{2L, 3L, 4L}}};
	D.ss<integer>("E") = 142;
	D.ss<timestamp>("now") = make_timestamp();
	test_json(D["E"]);
	test_json(D);
}

