/// @file
/// @author uentity
/// @date 29.06.2018
/// @brief Test cases for BS tree
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#define BOOST_TEST_DYN_LINK

#include "test_objects.h"
#include <bs/kernel.h>
#include <bs/kernel_tools.h>
#include <bs/log.h>
#include <bs/tree.h>

#include <bs/serialize/base_types.h>
#include <bs/serialize/array.h>
#include <bs/serialize/tree.h>

#include <boost/test/unit_test.hpp>
#include <iostream>

#define K BS_KERNEL

namespace {

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

} // eof hidden namespace

BOOST_AUTO_TEST_CASE(test_tree) {
	using namespace blue_sky;
	using namespace blue_sky::log;
	using namespace blue_sky::tree;

	std::cout << "*********************************************************************" << std::endl;

	// person
	sp_obj P = BS_KERNEL.create_object(bs_person::bs_type(), std::string("Tyler"), double(33));
	// person link
	auto L = std::make_shared<hard_link>("person link", std::move(P));
	BOOST_TEST(L);
	auto L1 = test_json(L);
	BOOST_TEST(L->name() == L1->name());

	// create root link and node
	sp_node N = K.create_object("node");
	// create several persons and insert 'em into node
	for(int i = 0; i < 10; ++i) {
		std::string p_name = "Citizen_" + std::to_string(i);
		N->insert(std::make_shared<hard_link>(
			p_name, K.create_object("bs_person", p_name, double(i + 20))
		));
	}
	// create hard link referencing first object
	N->insert(std::make_shared<hard_link>(
		"hard_Citizen_0", N->begin()->get()->data()
	));
	// create weak link referencing 2nd object
	N->insert(std::make_shared<weak_link>(
		"weak_Citizen_1", N->find(1)->get()->data()
	));
	// create sym link referencing 3rd object
	N->insert(std::make_shared<sym_link>(
		"sym_Citizen_2", abspath(*N->find(2))
	));
	// print resulting tree content
	kernel_tools::print_link(std::make_shared<hard_link>("r", N));

	// serializze node
	auto N1 = test_json(N);
	// print loaded tree content
	kernel_tools::print_link(std::make_shared<hard_link>("r", N1));
	BOOST_TEST(N1);
	BOOST_TEST(N1->size() == N->size());
}

