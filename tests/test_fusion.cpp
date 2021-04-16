/// @date 17.08.2020
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#define BOOST_TEST_DYN_LINK

#include "test_objects.h"
#include "test_serialization.h"

#include <bs/actor_common.h>
#include <bs/log.h>
#include <bs/propdict.h>
#include <bs/kernel/radio.h>
#include <bs/kernel/types_factory.h>
#include <bs/kernel/tools.h>

#include <bs/tree/fusion.h>
#include <bs/tree/tree.h>

#include <iostream>
#include <atomic>
#include <chrono>
#include <thread>

using namespace blue_sky;
using namespace std::string_literals;
using Req = tree::Req;
using ReqStatus = tree::ReqStatus;

inline constexpr auto level_width = 10;
inline constexpr auto obj_level = 3;

// simple fusion bridge
struct test_bridge : tree::fusion_iface {

	auto depth(tree::link L) const {
		int res = 0;
		while((L = tree::owner_handle(L)))
			++res;
		return res;
	}

	auto do_pull_data(sp_obj root, tree::link root_lnk, prop::propdict) -> error override {
		return perfect;
	}

	auto do_populate(sp_obj root, tree::link root_lnk, prop::propdict) -> error override {
		bsout() << "<- populating [{}, {}]" <<
			to_string(root_lnk.id()) << root_lnk.name(unsafe) << bs_end;

		auto N = root->data_node();
		if(!N) return tree::Error::NotANode;
		const auto D = depth(root_lnk);

		for(int i = 0; i < level_width; ++i) {
			std::string p_name = std::to_string(i);
			if(D < obj_level)
				N.insert(tree::fusion_link(p_name, tree::node()));
			else {
				p_name = "Citizen_" + p_name;
				N.insert(tree::fusion_link{
					p_name, kernel::tfactory::create_object("bs_person", p_name, double(i + 20))
				});
			}
		}
		return perfect;
	}
};

// async tree loader
auto fetched_count() -> std::atomic<int>& {
	static std::atomic<int> cnt_ = 0;
	return cnt_;
}

auto fetch_tree(tree::node_or_err N, tree::link lnk) {
	using namespace blue_sky;

	bsout() << "-> fetch_tree [D = {}, DN = {}] {}: valid = {}, is nill = {}, size = {}" <<
		int(lnk.req_status(Req::Data)) << int(lnk.req_status(Req::DataNode)) <<
		lnk.name(unsafe) << bool(N) <<
		(N ? N->is_nil() : true) << (N ? N->size() : 0) << std::endl;

	if((!N && N.error().code == tree::Error::NotANode) || N->is_nil()) {
		++fetched_count();
		return;
	}

	//for(auto& child_lnk : N->leafs())
	//	child_lnk.data_node(fetch_tree);

	anon_request(
		N->actor(), kernel::radio::timeout(), false, [](const tree::links_v& leafs) {
			bsout() << "-> fetch_tree::leafs size = {}" << leafs.size() << bs_end;
			for(auto& child_lnk : leafs)
				child_lnk.data_node(fetch_tree);
		}, a_node_leafs(), tree::Key::AnyOrder
	);
}

BOOST_AUTO_TEST_CASE(test_fusion) {
	std::cout << "\n\n*** testing tree events..." << std::endl;
	std::cout << "*********************************************************************" << std::endl;

	auto r = tree::link::make_root<tree::fusion_link>("/", tree::node(), std::make_shared<test_bridge>());
	r.data_node(fetch_tree);

	while(true) {
		std::this_thread::sleep_for(std::chrono::milliseconds(500));
		int cnt = fetched_count();
		bsout() << "... fetched {} persons" << cnt << bs_end;
		if(cnt == 10000) break;
	}
}
