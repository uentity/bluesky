/// @file
/// @author uentity
/// @date 17.07.2019
/// @brief Test messages passing inside tree
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#define BOOST_TEST_DYN_LINK

#include "test_objects.h"
#include "test_serialization.h"

#include <bs/log.h>
#include <bs/propdict.h>
#include <bs/kernel/kernel.h>
#include <bs/kernel/tools.h>
#include <bs/tree/tree.h>

#include <bs/serialize/base_types.h>
#include <bs/serialize/array.h>
#include <bs/serialize/tree.h>

#include <boost/uuid/uuid_io.hpp>
#include <boost/test/unit_test.hpp>
#include <caf/scoped_actor.hpp>

#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>

using namespace std::chrono_literals;
using namespace blue_sky;
using namespace blue_sky::log;
using namespace blue_sky::tree;
using namespace blue_sky::prop;

BOOST_AUTO_TEST_CASE(test_tree_events) {
	std::cout << "\n\n*** testing tree events..." << std::endl;
	std::cout << "*********************************************************************" << std::endl;
	//// person
	//sp_obj P = kernel::tfactory::create_object(bs_person::bs_type(), std::string("Tyler"), double(33));
	//// person link
	//auto L = std::make_shared<hard_link>("person link", std::move(P));

	auto hN = make_persons_tree();
	auto N = hN->data_node();
	auto L = N->find("hard_Citizen_0", Key::Name);
	// make deeper tree
	auto N1 = std::make_shared<node>();
	N1->insert("N", N);
	auto N2 = std::make_shared<node>();
	N2->insert("N1", N1);
	auto N3 = std::make_shared<node>();
	N3->insert("N2", N2);

	// test link rename

	auto test_rename = [](auto tgt, sp_link src) -> int {
		using T = decltype(tgt);
		using P = std::add_const_t<typename T::element_type>;

		std::atomic<int> rename_cnt = 0;
		auto rename_cb = [&](std::shared_ptr<P> who, Event, prop::propdict what) -> void {
			++rename_cnt;
			//bsout() << "=> {}: renamed '{}' -> '{}'" << to_string(who->id())
			//	<< get<std::string>(what, "prev_name") << get<std::string>(what, "new_name") << bs_end;
		};

		auto h_rename = tgt->subscribe(rename_cb, Event::LinkRenamed);
		for(int i = 0; i < 1000; ++i)
			src->rename("Tyler " + std::to_string(i));
		std::this_thread::sleep_for(200ms);

		tgt->unsubscribe(h_rename);
		std::this_thread::sleep_for(10ms);

		for(int i = 0; i < 100; ++i)
			src->rename("Tyler " + std::to_string(i));
		std::this_thread::sleep_for(200ms);
		return rename_cnt;
	};

	std::cout << "### rename link->node calls: " << test_rename(N3, L) << std::endl;
	std::cout << "### rename link->link calls: " << test_rename(L, L) << std::endl;

	/////////////////////////////////////////////////////////////////////////////////
	//  status
	//

	auto test_status = [](auto tgt, sp_link src) -> int {
		using T = decltype(tgt);
		using P = std::add_const_t<typename T::element_type>;

		std::atomic<int> status_cnt = 0;
		auto status_cb = [&](std::shared_ptr<P> who, Event, prop::propdict what) {
			++status_cnt;
			//bsout() << "=> {}: status {}: {} -> {}" << to_string(who->id()) <<
			//	get<integer>(what, "request") << get<integer>(what, "prev_status") << 
			//	get<integer>(what, "new_status") << bs_end;
		};

		auto h_status = tgt->subscribe(status_cb, Event::LinkStatusChanged);
		for(int i = 0; i < 1000; ++i)
			src->rs_reset(Req::Data, ReqStatus((int)ReqStatus::OK + i & 1));
		std::this_thread::sleep_for(200ms);

		tgt->unsubscribe(h_status);
		std::this_thread::sleep_for(10ms);

		for(int i = 0; i < 100; ++i)
			src->rs_reset(Req::Data, ReqStatus((int)ReqStatus::OK + i & 1));
		std::this_thread::sleep_for(200ms);
		return status_cnt;
	};

	// summary
	std::cout << "### status link->node calls: " << test_status(N3, L) << std::endl;
	std::cout << "### status link->link calls: " << test_status(L, L) << std::endl;
}
