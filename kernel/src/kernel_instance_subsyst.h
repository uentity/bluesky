/// @file
/// @author uentity
/// @date 15.03.2017
/// @brief Manage instances of BlueSky types stored inside kernel
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include <bs/kernel.h>
#include <set>
#include <mutex>
//#include <boost/pool/pool_alloc.hpp>

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(detail)

struct BS_HIDDEN_API kernel_instance_subsyst {
	using instances_enum = kernel::instances_enum;

	using instances_storage_t = std::set<
		sp_cobj
		//boost::fast_pool_allocator<
		//	sp_obj, boost::default_user_allocator_new_delete, boost::details::pool::null_mutex
		//>
	>;
	using instances_map_t = std::unordered_map< BS_TYPE_INFO, instances_storage_t >;
	instances_map_t instances_;
	// sync access to instances
	std::mutex solo_;

	//register instance of any BlueSky type
	int register_instance(sp_cobj&& obj);

	int free_instance(sp_cobj&& obj);

	instances_enum instances(const BS_TYPE_INFO& ti) const;
};

NAMESPACE_END(detail)
NAMESPACE_END(blue_sky)

