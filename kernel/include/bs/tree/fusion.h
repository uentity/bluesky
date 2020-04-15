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
 *  Interface of Fusion client that have to populate root item with children
 *-----------------------------------------------------------------------------*/
class BS_API fusion_iface {
public:
	/// accept root object and optionally type of child objects to be populated with
	virtual auto populate(const sp_node& root, const std::string& child_type_id = "") -> error = 0;
	/// download passed object's content from third-party backend
	virtual auto pull_data(const sp_obj& root) -> error = 0;

	virtual ~fusion_iface();
};
using sp_fusion = std::shared_ptr<fusion_iface>;

/*-----------------------------------------------------------------------------
 *  Fusion link populates object children when `data_node()` or `data()` is called
 *-----------------------------------------------------------------------------*/
class BS_API fusion_link : public link {
	friend class blue_sky::atomizer;
	friend class cereal::access;

public:
	using super = link;

	using fusion_actor_type = caf::typed_actor<
		// populate pointee with given children types
		caf::replies_to<a_flnk_populate, std::string, bool>::with<result_or_errbox<sp_node>>,
		// get link's bridge
		caf::replies_to<a_flnk_bridge>::with<sp_fusion>,
		// set link's bridge
		caf::reacts_to<a_flnk_bridge, sp_fusion>
	>;

	using actor_type = link::actor_type::extend_with<fusion_actor_type>;

	// ctors
	fusion_link(
		std::string name, sp_node data = nullptr,
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
		-> result_or_err<sp_node>;
	// async populate
	auto populate(process_data_cb f, std::string child_type_id) const
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
