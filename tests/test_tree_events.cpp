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
#include <unordered_map>

using namespace std::chrono_literals;
using namespace blue_sky;
using namespace blue_sky::log;
using namespace blue_sky::tree;
using namespace blue_sky::prop;

BOOST_AUTO_TEST_CASE(test_tree_events) {
	std::cout << "\n\n*** testing tree events..." << std::endl;
	std::cout << "*********************************************************************" << std::endl;

	// setup counters
	using counters_t = std::unordered_map<Event, std::atomic<int>>;
	auto counters = std::make_shared<counters_t>();
	(*counters)[Event::LinkInserted] = 0;
	(*counters)[Event::LinkErased] = 0;
	(*counters)[Event::LinkDeleted] = 0;
	(*counters)[Event::LinkRenamed] = 0;
	(*counters)[Event::LinkStatusChanged] = 0;

	// setup events processor
	auto ev_processor_cb = [=](event ev) -> void {
		const auto ev_to_string = [&]() -> std::string_view {
			switch(ev.code) {
			case Event::LinkInserted: return "LinkInserted";
			case Event::LinkErased: return "LinkErased";
			case Event::LinkDeleted: return "LinkDeleted";
			case Event::LinkRenamed: return "LinkRenamed";
			case Event::LinkStatusChanged: return "LinkStatusChanged";
			default: return "<unknown event>";
			};
		};

		++(*counters)[ev.code];
		auto origin_tid = [&] {
			if(auto L = ev.origin_link())
				return L.type_id();
			else
				return ev.origin_node().type_id();
		}();
		bsout() << "=> {}.{}: {}" << origin_tid << ev_to_string() << to_string(ev.params) << bs_end;
	};

	static const auto adapt2node = [](auto cb) {
		return [cb = std::move(cb)](auto /*subnode*/, event ev) {
			cb(std::move(ev));
		};
	};

	auto hN = make_persons_tree();
	auto N = hN.data_node();
	auto L = N.find("hard_Citizen_0", Key::Name);

	// make deeper tree
	auto N1 = node();
	N1.subscribe(adapt2node(ev_processor_cb), Event::All);
	N1.insert("N", node());
	N1.find("N", Key::Name).data_node().insert("N", node());
	N1.find("N", Key::Name).data_node().find("N", Key::Name).data_node().insert("N", node());
	std::this_thread::sleep_for(200ms);

	// test link rename
	auto test_rename = [](auto&& tgt, tree::link src) -> int {
		std::atomic<int> rename_cnt = 0;
		auto rename_cb = [&](event ev) -> void {
			++rename_cnt;
			//bsout() << "=> {}.{}: {}" << who->type_id() << 
			//	get<std::string>(what, "prev_name") << get<std::string>(what, "new_name") << bs_end;
		};

		auto h_rename = tgt.subscribe([&] {
			if constexpr(std::is_same_v<std::decay_t<decltype(tgt)>, node>)
				return adapt2node(rename_cb);
			else
				return rename_cb;
		}(), Event::LinkRenamed);
		for(int i = 0; i < 1000; ++i)
			src.rename("Tyler " + std::to_string(i));
		std::this_thread::sleep_for(200ms);

		tgt.unsubscribe(h_rename);
		std::this_thread::sleep_for(10ms);

		for(int i = 0; i < 100; ++i)
			src.rename("Tyler " + std::to_string(i));
		std::this_thread::sleep_for(200ms);
		return rename_cnt;
	};

	std::cout << "### rename link->node calls: " << test_rename(N1, L) << std::endl;
	std::cout << "### rename link->link calls: " << test_rename(L, L) << std::endl;

	/////////////////////////////////////////////////////////////////////////////////
	//  status
	//

	auto test_status = [](auto&& tgt, tree::link src) -> int {
		using T = std::decay_t<decltype(tgt)>;

		std::atomic<int> status_cnt = 0;
		auto status_cb = [&](event ev) {
			++status_cnt;
			//bsout() << "=> {}: status {}: {} -> {}" << to_string(who->id()) <<
			//	get<integer>(what, "request") << get<integer>(what, "prev_status") << 
			//	get<integer>(what, "new_status") << bs_end;
		};

		auto h_status = tgt.subscribe([&] {
			if constexpr(std::is_same_v<std::decay_t<decltype(tgt)>, node>)
				return adapt2node(status_cb);
			else
				return status_cb;
		}(), Event::LinkStatusChanged);
		for(int i = 0; i < 1000; ++i)
			src.rs_reset(Req::Data, ReqStatus((int)ReqStatus::OK + i & 1));
		std::this_thread::sleep_for(200ms);

		tgt.unsubscribe(h_status);
		std::this_thread::sleep_for(10ms);

		for(int i = 0; i < 100; ++i)
			src.rs_reset(Req::Data, ReqStatus((int)ReqStatus::OK + i & 1));
		std::this_thread::sleep_for(200ms);
		return status_cnt;
	};

	// summary
	std::cout << "### status link->node calls: " << test_status(N1, L) << std::endl;
	std::cout << "### status link->link calls: " << test_status(L, L) << std::endl;
}
