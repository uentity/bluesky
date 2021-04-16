/// @file
/// @author uentity
/// @date 10.08.2018
/// @brief Fusion link implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "fusion_link_actor.h"

#include <bs/log.h>
#include <bs/kernel/types_factory.h>

#include <memory_resource>

#define FIMPL static_cast<fusion_link_impl&>(*pimpl())

NAMESPACE_BEGIN(blue_sky::tree)
using ei = engine::impl;

// setup synchronized pool allocator for link impls
static auto impl_pool = std::pmr::synchronized_pool_resource{};
static auto fusion_impl_alloc = std::pmr::polymorphic_allocator<fusion_link_impl>(&impl_pool);

/*-----------------------------------------------------------------------------
 *  fusion_iface
 *-----------------------------------------------------------------------------*/
auto fusion_iface::is_uniform(const sp_obj&) const -> bool {
	// [NOTE] assume that objects are usually uniform
	return true;
}

auto fusion_iface::pull_data(sp_obj root, link root_link, prop::propdict params) -> error {
	return error::eval_safe([&] {
		do_pull_data(std::move(root), std::move(root_link), std::move(params));
	});
}

auto fusion_iface::populate(sp_obj root, link root_link, prop::propdict params) -> error {
	// check precondition: passed object contains valid node
	if(root->data_node())
		return error::eval_safe([&] {
			do_populate(std::move(root), std::move(root_link), std::move(params));
		});
	else
		return error::quiet(Error::NotANode);
}

/*-----------------------------------------------------------------------------
 *  fusion_link
 *-----------------------------------------------------------------------------*/
fusion_link::fusion_link(
	std::string name, sp_obj data, sp_fusion bridge, Flags f
) : // set LazyLoad flag by default
	super(std::allocate_shared<fusion_link_impl>(
		fusion_impl_alloc, std::move(name), std::move(data), std::move(bridge), Flags(f | LazyLoad)
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
	: super(std::allocate_shared<fusion_link_impl>(fusion_impl_alloc), false)
{}

auto fusion_link::pull_data(prop::propdict params, bool wait_if_busy) const -> obj_or_err {
	return pimpl()->actorf<obj_or_errbox>(
		*this, a_flnk_data(), std::move(params), wait_if_busy
	);
}

auto fusion_link::pull_data(link::process_data_cb f, prop::propdict params) const -> void {
	using result_t = obj_or_errbox;

	anon_request<caf::detached>(
		actor(*this), kernel::radio::timeout(true), false,
		[f = std::move(f), self = *this](result_t data) mutable {
			f( std::move(data), std::move(self) );
		},
		a_flnk_data(), std::move(params), true
	);
}

auto fusion_link::populate(prop::propdict params, bool wait_if_busy) const -> node_or_err {
	return unpack(pimpl()->actorf<node_or_errbox>(
		*this, a_flnk_populate(), std::move(params), wait_if_busy
	));
}

auto fusion_link::populate(link::process_dnode_cb f, prop::propdict params) const -> void {
	using result_t = node_or_errbox;

	anon_request<caf::detached>(
		actor(*this), kernel::radio::timeout(true), false,
		[f = std::move(f), self = *this](result_t data) mutable {
			f( unpack(std::move(data)), std::move(self) );
		},
		a_flnk_populate(), std::move(params), true
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
