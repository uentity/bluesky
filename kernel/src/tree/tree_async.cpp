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

NAMESPACE_BEGIN(blue_sky::tree) NAMESPACE_BEGIN(detail)

/*-----------------------------------------------------------------------------
 *  implementation details
 *-----------------------------------------------------------------------------*/
NAMESPACE_BEGIN()
template<typename level_process_f>
struct deref_actor : blue_sky::detail::async_api_mixin<deref_actor<level_process_f>> {
	using base_t = blue_sky::detail::async_api_mixin<deref_actor<level_process_f>>;
	using deref_actor_t = caf::typed_actor<
		caf::reacts_to<std::string, sp_link, level_process_f, deref_process_f, bool>
	>;

	deref_actor_t actor_;
	auto actor() const -> const deref_actor_t& { return actor_; }

	deref_actor()
		: actor_(blue_sky::kernel::config::actor_system().spawn(behaviour))
	{
		base_t::init_sender();
	}

private:
	static auto behaviour(typename deref_actor_t::pointer self) -> typename deref_actor_t::behavior_type {
		return {
			[](
				const std::string& path, sp_link lnk,
				const level_process_f& lp, const deref_process_f& f,
				bool follow_lazy_links
			) {
				f(deref_path_impl(path, std::move(lnk), nullptr, follow_lazy_links, lp));
			}
		};
	}

};

NAMESPACE_END()

template<typename DerefProcessF>
auto deref_path_async(
	std::string path, sp_link lnk, walk_down_ft&& lp, DerefProcessF&& dp,
	bool follow_lazy_links, bool high_priority
) -> void {
	// initialize actor only once
	static deref_actor<walk_down_ft> actor;

	// sanity
	if(!lnk) dp(nullptr);
	// send message
	actor.send(
		high_priority ? caf::message_priority::high : caf::message_priority::normal,
		std::move(path), std::move(lnk), std::move(lp), std::move(dp),
		follow_lazy_links
	);
}

NAMESPACE_END(detail)

/*-----------------------------------------------------------------------------
 *  async tree fucntions
 *-----------------------------------------------------------------------------*/
// same as above but accept std::function
auto deref_path(
	deref_process_f f, std::string path, sp_link start, node::Key path_unit,
	bool follow_lazy_links, bool high_priority
) -> void {
	detail::deref_path_async<deref_process_f>(
		std::move(path), std::move(start),
		detail::gen_walk_down_tree(path_unit), std::move(f),
		follow_lazy_links, high_priority
	);
}

NAMESPACE_END(blue_sky::tree)

