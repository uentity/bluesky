/// @file
/// @author uentity
/// @date 09.08.2018
/// @brief Fusion link and Fusion client interface declaration
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "link.h"

NAMESPACE_BEGIN(blue_sky::tree)

/*-----------------------------------------------------------------------------
 *  Interface of Fusion brigde that implements filling target item
 *-----------------------------------------------------------------------------*/
class BS_API fusion_iface {
public:
	/// accept root object and optionally type of child objects to be populated with
	auto populate(sp_obj root, link root_link, const std::string& child_type_id = "") -> error;
	/// download passed object's content from third-party backend
	auto pull_data(sp_obj root, link root_link) -> error;

	virtual ~fusion_iface() = default;

private:
	/// implementation to be overriden in derived fusion bridges
	virtual auto do_populate(sp_obj root, link root_link, const std::string& child_type_id) -> error = 0;
	virtual auto do_pull_data(sp_obj root, link root_link) -> error = 0;
};
using sp_fusion = std::shared_ptr<fusion_iface>;

/*-----------------------------------------------------------------------------
 *  Fusion link populates object children when `data_node()` or `data()` is called
 *-----------------------------------------------------------------------------*/
struct fusion_link_impl;

class BS_API fusion_link : public link {
	friend class blue_sky::atomizer;
	friend class cereal::access;

public:
	using super = link;
	using engine_impl = fusion_link_impl;

	using fusion_actor_type = caf::typed_actor<
		// populate pointee with given children types
		caf::replies_to<a_flnk_populate, std::string, bool>::with<node_or_errbox>,
		// get link's bridge
		caf::replies_to<a_flnk_bridge>::with<sp_fusion>,
		// set link's bridge
		caf::reacts_to<a_flnk_bridge, sp_fusion>
	>;

	using actor_type = link::actor_type::extend_with<fusion_actor_type>;

	// ctors
	fusion_link(
		std::string name, sp_obj data = nullptr,
		sp_fusion bridge = nullptr, Flags f = Plain
	);

	fusion_link(
		std::string name, const char* obj_type, std::string oid = "",
		sp_fusion bridge = nullptr, Flags f = Plain
	);
	/// convert from base link
	fusion_link(const link& rhs);

	static auto type_id_() -> std::string_view;

	// force `fusion_iface::populate()` call with specified children types
	// regardless of populate status
	auto populate(const std::string& child_type_id, bool wait_if_busy = true) const
	-> node_or_err;
	// async populate
	auto populate(process_dnode_cb f, std::string child_type_id) const
	-> void;

	// access to link's fusion bridge
	auto bridge() const -> sp_fusion;
	auto reset_bridge(sp_fusion new_bridge = nullptr) -> void;

private:
	// don't start internal actor
	fusion_link();
};

NAMESPACE_END(blue_sky::tree)

// support for hashed container of links
STD_HASH_BS_LINK(fusion_link)
