/// @file
/// @author uentity
/// @date 31.10.2018
/// @brief Async tree functions
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/tree/tree.h>
#include <bs/kernel/config.h>
#include <bs/detail/async_api_mixin.h>
#include <bs/serialize/tree.h>
#include <bs/serialize/cafbind.h>

#include "tree_impl.h"

#include <caf/all.hpp>

CAF_ALLOW_UNSAFE_MESSAGE_TYPE(blue_sky::tree::deref_process_f)

using walk_down_ft = decltype( blue_sky::tree::detail::gen_walk_down_tree() );
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(walk_down_ft)

NAMESPACE_BEGIN(blue_sky::tree)

/*-----------------------------------------------------------------------------
 *  implementation details
 *-----------------------------------------------------------------------------*/
NAMESPACE_BEGIN()

template <typename level_process_f>
using deref_actor_t = caf::typed_actor<
	caf::reacts_to<std::string, sp_link, level_process_f, deref_process_f, bool>
>;

template<typename level_process_f>
struct deref_actor : blue_sky::detail::anon_async_api_mixin< deref_actor_t<level_process_f> > {
	using actor_t = deref_actor_t<level_process_f>;
	using base_t = blue_sky::detail::anon_async_api_mixin<actor_t>;

	// lazy init actor only when message arrives
	// and don't auto-kill actor when this dies
	deref_actor() : base_t(false) {
		base_t::template spawn<caf::spawn_options::lazy_init_flag>(async_behavior);
	}

	static auto async_behavior(typename actor_t::pointer self) -> typename actor_t::behavior_type {
		return {
			[](
				const std::string& path, sp_link lnk,
				const level_process_f& lp, const deref_process_f& f,
				bool follow_lazy_links
			) {
				f(detail::deref_path_impl(path, std::move(lnk), nullptr, follow_lazy_links, lp));
			}
		};
	}

};

NAMESPACE_END()

/*-----------------------------------------------------------------------------
 *  async tree fucntions
 *-----------------------------------------------------------------------------*/
// same as above but accept std::function
auto deref_path(
	deref_process_f f, std::string path, sp_link start, node::Key path_unit,
	bool follow_lazy_links, bool high_priority
) -> void {
	// create local temp actor
	deref_actor<walk_down_ft> actor;

	// send message
	actor.send(
		high_priority ? caf::message_priority::high : caf::message_priority::normal,
		std::move(path), std::move(start), detail::gen_walk_down_tree(path_unit), std::move(f),
		follow_lazy_links
	);
	// and forget about actor
	// 1. message should arrive, then actor initialization is performed, then it gets executed
	// 2. and after that actor itself dies because no more strong references to it exist
}

NAMESPACE_END(blue_sky::tree)

