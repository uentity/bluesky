/// @date 24.08.2020
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "node_actor.h"
#include "node_extraidx_actor.h"

#include <bs/kernel/radio.h>
#include <bs/detail/scope_guard.h>

#include <algorithm>

#include "actor_debug.h"

NAMESPACE_BEGIN(blue_sky::tree::detail)
/*-----------------------------------------------------------------------------
 *  this impl does no blocked waits and suitable for direct invoke from node actor
 *-----------------------------------------------------------------------------*/
template<Key K = Key::ID>
auto deep_search_impl(
	node_actor* self, const Key_type<K>& key, bool return_first, lids_v active_symlinks = {}
) -> caf::result<links_v> {
	using namespace kernel::radio;

	// 0. prepare result promise
	auto res_promise = self->make_response_promise<links_v>();

	// 3. sequental leafs processor as recursive lambda
	auto do_search_leafs = [=](auto fimpl, auto work, links_v res, lids_v asl) mutable {
		// 1st possible outcome: end of work processing chain - deliver current result
		if(work.empty()) {
			adbg(self) << "DS: 3. deliver result (work is empty), res size = " << res.size() << std::endl;
			res_promise.deliver(std::move(res));
			return;
		}

		// cut current work
		auto L = work.back();
		work.pop_back();
		adbg(self) << "DS: 3. process leaf " << L.id() << std::endl;

		// 2nd possible outcome: recursion step to process remaining work
		auto make_next_iteration =
		[=, fimpl = std::move(fimpl), work = std::move(work)](links_v res) mutable {
			return [=, res = std::move(res)]() mutable {
				adbg(self) << "DS: 3. recursion to next step" << std::endl;
				// invoke self with rest of work
				fimpl(fimpl, std::move(work), std::move(res), std::move(asl));
			};
		};
		// fallback next step if error is encountered on current leaf
		auto next_step = make_next_iteration(res);

		// remember symlink or quit if we meet already processed links
		const auto is_symlink = L.type_id() == sym_link::type_id_();
		if(const auto Lid = L.id(); is_symlink) {
			auto pos = std::lower_bound(asl.begin(), asl.begin(), Lid);
			if(pos != asl.end()) {
				if(*pos == Lid) {
					// current work is already processed symlink - quit
					next_step();
					return;
				}
				else ++pos;
			}
			asl.insert(pos, Lid);
		}

		// functor that process one child leaf
		auto search_leaf = [=, res = std::move(res), asl = std::move(asl)]() mutable {
			adbg(self) << "DS: 3. DataNode request to link " << L.id() << std::endl;
			self->request(L.actor(), timeout(true), a_data_node{}, true)
			.then(
				[=, res = std::move(res), asl = std::move(asl)](const node_or_errbox& maybe_N) mutable {
					adbg(self) << "DS: 3. nested node = " << (bool)maybe_N <<
						", res.size = " << res.size() << std::endl;
					if(!maybe_N) {
						next_step();
						return;
					}

					// use specific request for ID key
					// get next level results
					[&] {
						const node& next_n = *maybe_N;
						adbg(self) << "DS: 3. DeepSearch request to nested node " << next_n.home_id() << std::endl;
						if constexpr(K == Key::ID)
							return self->request(
								node_impl::actor(next_n), timeout(true),
								a_node_deep_search(), key, std::move(asl)
							);
						else
							return self->request(
								node_impl::actor(next_n), timeout(true),
								a_node_deep_search(), key, K, return_first, std::move(asl)
							);
					}().then(
						[=, res = std::move(res)](const links_v& next_l) mutable {
							adbg(self) << "DS: 3. process result of DeepSearch request, count = " <<
								next_l.size() << std::endl;
							// merge new results into existing
							std::copy(next_l.begin(), next_l.end(), std::back_inserter(res));
							// deliver result or jump to next step
							if(return_first && !res.empty())
								res_promise.deliver(links_v{res.front()});
							else
								make_next_iteration(std::move(res))();
						},
						// goto next iteration on error
						[=](const caf::error&) mutable { next_step(); }
					);
				},
				// goto next iteration on error
				[=](const caf::error&) mutable { next_step(); }
			);
		};

		// check populated status before moving to next level
		if(L.req_status(Req::DataNode) == ReqStatus::OK)
			search_leaf();
		else {
			adbg(self) << "DS: 3. need to check flags of link " << L.id() << std::endl;
			// we have to check L's flags to not expand lazy links
			self->request(L.actor(), timeout(), a_lnk_flags())
			.then(
				[=, search_leaf = std::move(search_leaf)](Flags Lf) mutable {
					// don't enter lazy load links
					if(Lf & LazyLoad)
						next_step();
					else
						search_leaf();
				},
				// goto next iteration on error
				[=](const caf::error&) mutable { next_step(); }
			);
		}
	};

	// 2. starts leafs processing after local search is done
	auto do_deep_search =
	[=, do_search_leafs = std::move(do_search_leafs), asl = std::move(active_symlinks)](links_v res) mutable {
		adbg(self) << "DS: 2. do_deep_search, res size = " << res.size() << std::endl;
		if(return_first && !res.empty()){
			res_promise.deliver(links_v{res.front()});
			return;
		}

		// get reverted leafs list because `do_search_leaf()` eats work from the back
		auto Ls = self->impl.values<Key::AnyOrder>();
		std::reverse(Ls.begin(), Ls.end());
		adbg(self) << "DS: 2. search in children, count = " << Ls.size() << std::endl;
		do_search_leafs(do_search_leafs, std::move(Ls), std::move(res), std::move(asl));
	};

	// 1. do direct search in leafs
	if constexpr(has_builtin_index(K)) {
		adbg(self) << "DS: 1. direct search in builtin index" << std::endl;
		do_deep_search(self->impl.equal_range<K>(key).extract_values());
	}
	else {
		adbg(self) << "DS: 1. direct search in non-builtin index" << std::endl;
		// for non-builtin indices do 'equal range' request, then proceed
		self->request(
			self->spawn(extraidx_search_actor), caf::infinite,
			a_node_equal_range(), key, K, self->impl.values<Key::AnyOrder>()
		).then(
			[self, do_deep_search = std::move(do_deep_search)](links_v res) mutable {
				do_deep_search(std::move(res));
			},
			// fail fast if we can't even search inside own leafs
			[=](const caf::error&) mutable { res_promise.deliver(links_v{}); }
		);
	}
	adbg(self) << "DS: 0. res promise returned" << std::endl;
	return res_promise;
}

NAMESPACE_END(blue_sky::tree::detail)
