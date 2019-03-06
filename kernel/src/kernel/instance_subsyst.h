/// @file
/// @author uentity
/// @date 15.03.2017
/// @brief Manage instances of BlueSky types stored inside kernel
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/common.h>
#include <set>
#include <map>
#include <mutex>

NAMESPACE_BEGIN(blue_sky::kernel::detail)

struct BS_HIDDEN_API instance_subsyst {
	using instances_enum = std::vector<sp_cobj>;

	using instances_storage_t = std::set<
		sp_cobj
		//boost::fast_pool_allocator<
		//	sp_obj, boost::default_user_allocator_new_delete, boost::details::pool::null_mutex
		//>
	>;
	using instances_map_t = std::map<
		std::string, instances_storage_t, std::less<>
	>;
	instances_map_t instances_;
	// sync access to instances
	std::mutex solo_;

	//register instance of any BlueSky type
	auto register_instance(sp_cobj&& obj) -> int;

	auto free_instance(sp_cobj&& obj) -> int;

	auto instances(std::string_view type_id) const -> instances_enum;
};

NAMESPACE_END(blue_sky::kernel::detail)
