/// @file
/// @author uentity
/// @date 31.10.2018
/// @brief Async tree functions
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/tree/tree.h>
#include <bs/kernel.h>
#include <bs/detail/async_api_mixin.h>
#include <bs/serialize/tree.h>
#include <bs/serialize/cafbind.h>

#include "tree_impl.h"

#include <caf/all.hpp>

CAF_ALLOW_UNSAFE_MESSAGE_TYPE(blue_sky::tree::deref_process_f)
// [TODO] solve problem with function pointer
//CAF_ALLOW_UNSAFE_MESSAGE_TYPE(blue_sky::tree::deref_process_fp)

using walk_down_ft = decltype( blue_sky::tree::detail::gen_walk_down_tree() );
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(walk_down_ft)

NAMESPACE_BEGIN(blue_sky) NAMESPACE_BEGIN(tree) NAMESPACE_BEGIN(detail)
using namespace blue_sky::detail;

/*-----------------------------------------------------------------------------
 *  implementation details
 *-----------------------------------------------------------------------------*/
NAMESPACE_BEGIN()
template<typename level_process_f>
struct deref_actor : detail::async_api_mixin<deref_actor<level_process_f>> {
	using base_t = detail::async_api_mixin<deref_actor<level_process_f>>;
	using deref_actor_t = caf::typed_actor<
		caf::reacts_to<std::string, sp_link, level_process_f, deref_process_f>
		//caf::reacts_to<std::string, sp_link, level_process_f, deref_process_fp>
	>;

	deref_actor_t actor_;
	auto actor() const -> const deref_actor_t& { return actor_; }

	deref_actor()
		: actor_(BS_KERNEL.actor_system().spawn(behaviour))
	{
		base_t::init_sender();
	}

private:
	static auto behaviour(typename deref_actor_t::pointer self) -> typename deref_actor_t::behavior_type {
		return {
			[](
				const std::string& path, const sp_link& lnk,
				const level_process_f& lp, const deref_process_f& f
			) {
				f(deref_path(path, *lnk, lp));
			}
			//[](
			//	const std::string& path, const sp_link& lnk,
			//	const level_process_f& lp, deref_process_fp fp
			//) {
			//	fp(deref_path(path, *lnk, lp));
			//}
		};
	}

};

NAMESPACE_END()

template<typename DerefProcessF>
auto deref_path_async(
	std::string path, sp_link lnk, walk_down_ft&& lp, DerefProcessF&& dp
) -> void {
	// initialize actor only once
	static deref_actor<walk_down_ft> actor;

	// sanity
	if(!lnk) dp(nullptr);
	// send message
	actor.send(std::move(path), std::move(lnk), std::move(lp), std::move(dp));
}

NAMESPACE_END(detail)

/*-----------------------------------------------------------------------------
 *  async tree fucntions
 *-----------------------------------------------------------------------------*/
// same as above but accept std::function
auto deref_path(deref_process_f f, std::string path, sp_link start, node::Key path_unit) -> void {
	detail::deref_path_async<deref_process_f>(
		std::move(path), std::move(start),
		detail::gen_walk_down_tree(path_unit), std::move(f)
	);
}
// accept function
//auto deref_path(deref_process_fp f, std::string path, sp_link start, node::Key path_unit) -> void {
//	detail::deref_path_async<deref_process_fp>(
//		std::move(path), std::move(start),
//		detail::gen_walk_down_tree(path_unit), f
//	);
//}

NAMESPACE_END(tree) NAMESPACE_END(blue_sky)
