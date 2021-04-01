/// @author Alexander Gagarin (@uentity)
/// @date 04.04.2021
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "../type_caf_id.h"
#include "link.h"
#include "fusion.h"
#include "node.h"

CAF_BEGIN_TYPE_ID_BLOCK(bs_tree, blue_sky::detail::bs_tree_cid_begin)

	CAF_ADD_TYPE_ID(bs_tree, (blue_sky::tree::Event))
	CAF_ADD_TYPE_ID(bs_tree, (blue_sky::tree::Flags))
	CAF_ADD_TYPE_ID(bs_tree, (blue_sky::tree::Key))
	CAF_ADD_TYPE_ID(bs_tree, (blue_sky::tree::InsertPolicy))
	CAF_ADD_TYPE_ID(bs_tree, (blue_sky::tree::TreeOpts))
	CAF_ADD_TYPE_ID(bs_tree, (blue_sky::tree::Req))
	CAF_ADD_TYPE_ID(bs_tree, (blue_sky::tree::ReqReset))
	CAF_ADD_TYPE_ID(bs_tree, (blue_sky::tree::ReqStatus))

	CAF_ADD_TYPE_ID(bs_tree, (blue_sky::tree::link))
	CAF_ADD_TYPE_ID(bs_tree, (blue_sky::tree::links_v))
	CAF_ADD_TYPE_ID(bs_tree, (blue_sky::tree::lids_v))
	CAF_ADD_TYPE_ID(bs_tree, (blue_sky::tree::sp_fusion))

	CAF_ADD_TYPE_ID(bs_tree, (blue_sky::tree::node))
	CAF_ADD_TYPE_ID(bs_tree, (blue_sky::tree::node::existing_index))
	CAF_ADD_TYPE_ID(bs_tree, (blue_sky::tree::node::insert_status))

	CAF_ADD_TYPE_ID(bs_tree, (blue_sky::tree::obj_or_errbox))
	CAF_ADD_TYPE_ID(bs_tree, (blue_sky::tree::link_or_errbox))
	CAF_ADD_TYPE_ID(bs_tree, (blue_sky::tree::node_or_errbox))
	CAF_ADD_TYPE_ID(bs_tree, (blue_sky::tree::event))
	CAF_ADD_TYPE_ID(bs_tree, (blue_sky::result_or_errbox<blue_sky::tree::inodeptr>))

CAF_END_TYPE_ID_BLOCK(bs_tree)

CAF_ALLOW_UNSAFE_MESSAGE_TYPE(blue_sky::tree::event)
