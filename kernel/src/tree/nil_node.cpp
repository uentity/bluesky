/// @file
/// @author uentity
/// @date 03.07.2020
/// @brief Nil node impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "nil_engine.h"
#include "nil_engine_impl.h"
#include "node_impl.h"

#include <bs/defaults.h>

NAMESPACE_BEGIN(blue_sky::tree)
///////////////////////////////////////////////////////////////////////////////
//  nil node actor
//
struct nil_node::self_actor : nil_engine_actor {
	using super = nil_engine_actor;
	using insert_status = node::insert_status;

	using super::super;

	auto name() const -> const char* override { return "nil_node actor"; }

	auto make_behavior() -> behavior_type override { return node::actor_type::behavior_type{
	
		[](a_home) -> caf::group { return {}; },
		[](a_home_id) -> std::string { return nil_oid; },
		[](a_node_handle) -> link { return link{}; },
		[](a_node_size) -> std::size_t { return 0; },
		[](a_clone, bool) { return node::nil(); },

		[](a_node_leafs, Key) -> links_v { return {}; },
		[](a_node_keys, Key) -> lids_v { return {}; },
		[](a_node_keys, Key, Key) -> std::vector<std::string> { return {}; },
		[](a_node_ikeys, Key) -> std::vector<std::size_t> { return {}; },

		[](a_node_find, lid_type) -> link { return link{}; },
		[](a_node_find, std::size_t) -> link { return link{}; },
		[](a_node_find, const std::string&, Key) -> link { return link{}; },

		[](a_node_deep_search, lid_type) -> links_v { return {}; },
		[](a_node_deep_search, const std::string&, Key, bool) -> links_v { return {}; },

		[](a_node_index, lid_type) -> existing_index { return {}; },
		[](a_node_index, const std::string&, Key) -> existing_index { return {}; },

		[](a_node_equal_range, const std::string&, Key) -> links_v { return {}; },

		[](a_node_insert, link, InsertPolicy) -> insert_status { return {{}, false}; },
		[](a_node_insert, link, std::size_t, InsertPolicy) -> insert_status { return {{}, false}; },
		[](a_node_insert, const links_v&, InsertPolicy) -> std::size_t { return 0; },

		[](a_node_erase, lid_type) -> std::size_t { return 0; },
		[](a_node_erase, std::size_t) -> std::size_t { return 0; },
		[](a_node_erase, const std::string&, Key) -> std::size_t { return 0; },
		[](a_node_erase, const lids_v&) -> std::size_t { return 0; },
		[](a_node_clear) -> std::size_t { return 0; },

		[](a_lnk_rename, lid_type,           const std::string&) -> std::size_t { return 0; },
		[](a_lnk_rename, std::size_t,        const std::string&) -> std::size_t { return 0; },
		[](a_lnk_rename, const std::string&, const std::string&) -> std::size_t { return 0; },

		[](a_node_rearrange, const std::vector<std::size_t>&) -> error::box { return error{Error::EmptyData}; },
		[](a_node_rearrange, const lids_v&) -> error::box { return error{Error::EmptyData}; },

		[](a_apply, const node_transaction&) -> tr_result::box { return error{blue_sky::Error::TrEmptyTarget}; }

	}.unbox(); }

	auto on_exit() -> void override {
		nil_node::reset();
	}
};

///////////////////////////////////////////////////////////////////////////////
//  nil node impl
//
struct nil_node::self_impl : nil_engine_impl<nil_node, node_impl> {
	ENGINE_TYPE_DECL
};

ENGINE_TYPE_DEF(nil_node::self_impl, "__nil_node__")

///////////////////////////////////////////////////////////////////////////////
//  nil node
//
auto nil_node::nil_engine() -> const engine& {
	return nil_node::self_impl::internals();
}

auto nil_node::pimpl() -> const engine::sp_engine_impl& {
	return nil_node::self_impl::internals().pimpl_;
}

auto nil_node::actor() -> const engine::sp_ahandle& {
	return nil_node::self_impl::internals().actor_;
}

auto nil_node::reset() -> void {
	nil_node::self_impl::internals().reset();
}

auto nil_node::stop(bool wait_exit) -> void {
	nil_node::self_impl::internals().stop(wait_exit);
}

NAMESPACE_END(blue_sky::tree)
