/// @author uentity
/// @date 15.08.2018
/// @brief All atoms that are used in BS are declared here
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <caf/type_id.hpp>

CAF_BEGIN_TYPE_ID_BLOCK(bs_atoms, first_custom_type_id)
///////////////////////////////////////////////////////////////////////////////
//  common
//
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_launch_async)
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_unsafe)
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_long_op)
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_deep)

///////////////////////////////////////////////////////////////////////////////
//  generic BS API
//
	// discover neighbourhood
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_hi)
	// used to inform others that I'm quit
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_bye)
	// used as 'acquired` tag
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_ack)
	// used to invoke some processing over an object/actor
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_apply)
	// indicate that operation is lazy (won't start immediately)
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_lazy)

	// get implementation part of link/node/etc...
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_impl)
	// get home group of entity
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_home)
	// get home group ID
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_home_id)
	// obtain data (retrive object)
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_data)
	// obtain data node (retrive node)
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_data_node)
	// object save/load from storage
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_load)
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_save)
	// subscription manage
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_subscribe)
	// ask to clone some object
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_clone)

///////////////////////////////////////////////////////////////////////////////
//  link API
//
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_lnk_id)
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_lnk_name)
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_lnk_rename)

	CAF_ADD_ATOM(bs_atoms, blue_sky, a_lnk_status)
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_lnk_oid)
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_lnk_otid)
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_lnk_inode)
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_lnk_flags)

	CAF_ADD_ATOM(bs_atoms, blue_sky, a_mlnk_fresh)

	// async invoke `fusion_link::populate()`
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_flnk_data)
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_flnk_populate)
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_flnk_bridge)

///////////////////////////////////////////////////////////////////////////////
//  node API
//
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_node_size)
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_node_leafs)
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_node_keys)
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_node_ikeys)
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_node_find)
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_node_index)
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_node_deep_search)
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_node_equal_range)
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_node_deep_equal_range)
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_node_insert)
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_node_erase)
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_node_clear)

	// query node's actor group ID
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_node_disconnect)
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_node_handle)
	CAF_ADD_ATOM(bs_atoms, blue_sky, a_node_rearrange)

CAF_END_TYPE_ID_BLOCK(bs_atoms)

namespace blue_sky {

/// shorter aliases for some generic atoms
using launch_async_t = a_launch_async;
inline constexpr auto launch_async = a_launch_async_v;

using unsafe_t = a_unsafe;
inline constexpr auto unsafe = a_unsafe_v;

using long_op_t = a_long_op;
inline constexpr auto long_op = a_long_op_v;

using deep_t = a_deep;
inline constexpr auto deep = a_deep_v;

namespace detail {
inline constexpr caf::type_id_t bs_subsyst_gap = 50;

/// Misc types in parent 'bs' namespace
inline constexpr auto bs_cid_begin = bs_subsyst_gap * (caf::id_block::bs_atoms::end/bs_subsyst_gap + 1);
inline constexpr auto bs_private_cid_begin = bs_cid_begin + bs_subsyst_gap;
inline constexpr auto bs_cid_end = bs_private_cid_begin + bs_subsyst_gap;

/// Subsystems type id begin/end
inline constexpr auto bs_props_cid_begin = bs_cid_end;
inline constexpr auto bs_props_cid_end = bs_props_cid_begin + bs_subsyst_gap;

inline constexpr auto bs_transaction_cid = bs_props_cid_end;
inline constexpr auto bs_tr_cid_begin = bs_transaction_cid + 1;
inline constexpr auto bs_tr_cid_end = bs_tr_cid_begin + bs_subsyst_gap;

inline constexpr auto bs_tree_cid_begin = bs_tr_cid_end;
inline constexpr auto bs_tree_cid_end = bs_tree_cid_begin + bs_subsyst_gap;

} // eof blue_sky::detail

inline constexpr auto first_plugin_type_id = detail::bs_tree_cid_end;

} /* namespace blue_sky */
