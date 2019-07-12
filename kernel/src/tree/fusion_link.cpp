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

NAMESPACE_BEGIN(blue_sky::tree)

// default destructor for fusion_iface
fusion_iface::~fusion_iface() = default;

/*-----------------------------------------------------------------------------
 *  fusion_link
 *-----------------------------------------------------------------------------*/
fusion_link::fusion_link(
	std::string name, sp_node data, sp_fusion bridge, Flags f
) : // set LazyLoad flag by default
	ilink(spawn_lactor<fusion_link_actor>( std::move(name), data, bridge, Flags(f | link::LazyLoad) ))
{}

fusion_link::fusion_link(
	std::string name, const char* obj_type, std::string oid, sp_fusion bridge, Flags f
) :
	fusion_link(
		std::move(name),
		kernel::tfactory::create_object(obj_type, std::move(oid)), std::move(bridge), f
	)
{
	if(!pimpl()->data_)
		bserr() << log::E("fusion_link: cannot create object of type '{}'! Empty link!") <<
			obj_type << log::end;
}

auto fusion_link::pimpl() const -> fusion_link_actor* {
	return static_cast<fusion_link_actor*>(link::pimpl());
}

auto fusion_link::clone(bool deep) const -> sp_link {
	auto res = std::make_shared<fusion_link>(
		name(),
		deep ? kernel::tfactory::clone_object(pimpl()->data_) : pimpl()->data_,
		pimpl()->bridge_, flags()
	);
	return res;
}

auto fusion_link::type_id() const -> std::string {
	return "fusion_link";
}

auto fusion_link::populate(const std::string& child_type_id, bool wait_if_busy) const
-> result_or_err<sp_node> {
	return pimpl()->populate(child_type_id, wait_if_busy);
}

auto fusion_link::populate(link::process_data_cb f, std::string child_type_id) const -> void {
	using result_t = result_or_errbox<sp_node>;

	pimpl()->request(
		aimpl_, pimpl()->timeout_, a_flnk_populate(), true
	).then(
		[f = std::move(f), self = shared_from_this()](result_t data) {
			f(std::move(data), std::move(self));
		}
	);
}

auto fusion_link::bridge() const -> sp_fusion {
	return pimpl()->bridge();
}

auto fusion_link::reset_bridge(sp_fusion new_bridge) -> void {
	pimpl()->reset_bridge(std::move(new_bridge));
}

auto fusion_link::propagate_handle() -> result_or_err<sp_node> {
	// set handle of cached node object to this link instance
	self_handle_node(pimpl()->data_);
	return pimpl()->data() ?
		result_or_err<sp_node>(pimpl()->data_) : tl::make_unexpected(Error::EmptyData);
}

auto fusion_link::cache() const -> sp_node {
	return pimpl()->data_;
}

NAMESPACE_END(blue_sky::tree)
