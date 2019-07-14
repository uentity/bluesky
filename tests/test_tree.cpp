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

using namespace blue_sky;
using namespace blue_sky::log;
using namespace blue_sky::tree;
using namespace std::chrono_literals;

using Req = link::Req;
using ReqStatus = link::ReqStatus;

namespace {

class fusion_client : public fusion_iface {
	auto populate(const sp_node& root, const std::string& child_type_id = "") -> error override {
		bsout() << "fusion_client::populate() called" << end;
		return error::quiet();
	}

	auto pull_data(const sp_obj& root) -> error override {
		bsout() << "fusion_client::pull_data() called" << end;
		return error::quiet();
	}
};

} // eof hidden namespace

BOOST_AUTO_TEST_CASE(test_tree) {
	std::cout << "\n\n*** testing tree..." << std::endl;
	std::cout << "*********************************************************************" << std::endl;

	// person
	sp_obj P = kernel::tfactory::create_object(bs_person::bs_type(), std::string("Tyler"), double(33));
	// person link
	auto L = std::make_shared<hard_link>("person link", std::move(P));
	BOOST_TEST(L);
	auto L1 = test_json(L, false);
	BOOST_TEST(L->name() == L1->name());

	// create root link and node
	sp_node N = kernel::tfactory::create_object("node");
	// create several persons and insert 'em into node
	for(int i = 0; i < 10; ++i) {
		std::string p_name = "Citizen_" + std::to_string(i);
		N->insert(std::make_shared<hard_link>(
			p_name, kernel::tfactory::create_object("bs_person", p_name, double(i + 20))
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
	N->insert(std::make_shared<sym_link>(
		"sym_Citizen_3", abspath( deref_path(abspath(*N->find(3), node::Key::Name), N, node::Key::Name) )
	));
	N->insert(std::make_shared<sym_link>(
		"sym_dot", "."
	));
	// print resulting tree content
	auto hN = link::make_root<hard_link>("r", N);
	bsout() << "root node abspath: {}" << abspath(hN) << bs_end;
	bsout() << "root node abspath: {}" << convert_path(abspath(hN), hN, node::Key::ID, node::Key::Name) << bs_end;
	bsout() << "sym_Citizen_2 abspath: {}" << convert_path(
		abspath(*N->find("sym_Citizen_2", node::Key::Name)), hN, node::Key::ID, node::Key::Name
	) << bs_end;
	kernel::tools::print_link(hN, false);

	// serializze node
	auto N1 = test_json(N);

	// print loaded tree content
	kernel::tools::print_link(std::make_shared<hard_link>("r", N1), false);
	BOOST_TEST(N1);
	BOOST_TEST(N1->size() == N->size());

	// serialize to FS
	bsout() << "\n===========================\n" << bs_end;
	save_tree(hN, "tree_fs/.data", TreeArchive::FS);
	load_tree("tree_fs/.data", TreeArchive::FS).map([](const sp_link& hN1) {
		kernel::tools::print_link(hN1, false);
	});

	// test async dereference
	deref_path([](const sp_link& lnk) {
		std::cout << "*** Async deref callback: link : " <<
		(lnk ? abspath(lnk, node::Key::Name) : "None") << ' ' <<
		lnk->obj_type_id() << ' ' << (void*)lnk->data().get() << std::endl;
	}, "hard_Citizen_0", hN, node::Key::Name);

	// test link events
	std::atomic<int> rename_cnt = 0;
	auto h_rename = L->subscribe(link::Event::Renamed, [&](sp_link who, prop::propdict what) -> void {
		using namespace blue_sky::prop;
		++rename_cnt;
		bsout() << "=> {}: renamed '{}' -> '{}'" << to_string(who->id())
			<< get<std::string>(what, "prev_name") << get<std::string>(what, "new_name") << bs_end;
	});

	std::atomic<int> status_cnt = 0;
	auto h_status = L->subscribe(link::Event::StatusChanged, [&](sp_link who, prop::propdict what) {
		using namespace blue_sky::prop;
		++status_cnt;
		bsout() << "=> {}: status {}: {} -> {}" << to_string(who->id()) <<
			get<integer>(what, "request") << get<integer>(what, "prev_status") << 
			get<integer>(what, "new_status") << bs_end;
	});

	bsout() << "\n===========================\n" << bs_end;
	// try rename
	L->rename("Test rename event");
	// try change status
	L->rs_reset(Req::Data, ReqStatus::Error);
	L->rs_reset(Req::Data, ReqStatus::OK);
	//std::this_thread::sleep_for(10ms);

	// disconnect renamer
	//L->unsubscribe(h_rename);
	//std::this_thread::sleep_for(10ms);
	L->rename("Can't see me");
	
	// disconnect status
	L->rs_reset(Req::Data, ReqStatus::Error);
	//std::this_thread::sleep_for(10ms);
	//L->unsubscribe(h_status);
	// nobody will know abut that
	//std::this_thread::sleep_for(10ms);
	L->rs_reset(Req::Data, ReqStatus::OK);

	std::this_thread::sleep_for(500ms);
	std::cout << "### rename calls: " << rename_cnt << std::endl;
	std::cout << "### status calls: " << status_cnt << std::endl;
}

