/// @file
/// @author uentity
/// @date 10.08.2018
/// @brief Fusion link implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/log.h>
#include <bs/kernel/types_factory.h>
#include "fusion_link_actor.h"

#define FIMPL static_cast<fusion_link_impl&>(*pimpl())

NAMESPACE_BEGIN(blue_sky::tree)
/*-----------------------------------------------------------------------------
 *  fusion_iface
 *-----------------------------------------------------------------------------*/
auto fusion_iface::is_uniform(const sp_obj&) const -> bool {
	// [NOTE] assume that objects are usually uniform
	return true;
}

auto fusion_iface::pull_data(sp_obj root, link root_link) -> error {
	return error::eval_safe([&] {
		do_pull_data(std::move(root), std::move(root_link));
	});
}

auto fusion_iface::populate(sp_obj root, link root_link, const std::string& child_type_id) -> error {
	// check precondition: passed object contains valid node
	if(root->data_node())
		return error::eval_safe([&] {
			do_populate(std::move(root), std::move(root_link), child_type_id);
		});
	else
		return Error::NotANode;
}

/*-----------------------------------------------------------------------------
 *  fusion_link
 *-----------------------------------------------------------------------------*/
fusion_link::fusion_link(
	std::string name, sp_obj data, sp_fusion bridge, Flags f
) : // set LazyLoad flag by default
	super(std::make_shared<fusion_link_impl>(
		std::move(name), std::move(data), std::move(bridge), Flags(f | LazyLoad)
	))
{}

fusion_link::fusion_link(
	std::string name, node folder, sp_fusion bridge, Flags f
) : // set LazyLoad flag by default
	fusion_link(
		std::move(name), std::make_shared<objnode>(std::move(folder)), std::move(bridge), Flags(f | LazyLoad)
	)
{}

fusion_link::fusion_link(
	std::string name, const char* obj_type, std::string oid, sp_fusion bridge, Flags f
) :
	fusion_link(
		std::move(name),
		kernel::tfactory::create_object(obj_type, std::move(oid)), std::move(bridge), f
	)
{
	if(!FIMPL.data_)
		throw error(fmt::format("fusion_link: cannot create object of type '{}'! Empty link!", obj_type));
}

fusion_link::fusion_link()
	: super(std::make_shared<fusion_link_impl>(), false)
{}

auto fusion_link::populate(const std::string& child_type_id, bool wait_if_busy) const
-> node_or_err {
	return pimpl()->actorf<node_or_errbox>(
		*this, a_flnk_populate(), child_type_id, wait_if_busy
	);
}

auto fusion_link::populate(link::process_dnode_cb f, std::string child_type_id) const -> void {
	using result_t = node_or_errbox;

	anon_request<caf::detached>(
		actor(*this), kernel::radio::timeout(true), false,
		[f = std::move(f), self = *this](result_t data) mutable {
			f( std::move(data), std::move(self) );
		},
		a_flnk_populate(), std::move(child_type_id), true
	);
}

auto fusion_link::bridge() const -> sp_fusion {
	return FIMPL.bridge();
}

auto fusion_link::reset_bridge(sp_fusion new_bridge) -> void {
	FIMPL.reset_bridge(std::move(new_bridge));
}

LINK_CONVERT_TO(fusion_link)
LINK_TYPE_DEF(fusion_link, fusion_link_impl, "fusion_link")

NAMESPACE_END(blue_sky::tree)
