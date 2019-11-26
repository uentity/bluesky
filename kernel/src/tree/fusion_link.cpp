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
	super(std::make_shared<fusion_link_impl>( std::move(name), data, bridge, Flags(f | link::LazyLoad) ))
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

fusion_link::fusion_link()
	: super(std::make_shared<fusion_link_impl>(), false)
{}

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

auto fusion_link::pimpl() const -> fusion_link_impl* {
	return static_cast<fusion_link_impl*>(super::pimpl());
}

auto fusion_link::populate(const std::string& child_type_id, bool wait_if_busy) const
-> result_or_err<sp_node> {
	return actorf<result_or_errbox<sp_node>>(
		factor_, a_flnk_populate(), child_type_id, wait_if_busy
	);
}

auto fusion_link::populate(link::process_data_cb f, std::string child_type_id) const -> void {
	using result_t = result_or_errbox<sp_node>;

	anon_request(
		actor_, def_timeout(true), false,
		[f = std::move(f), self = shared_from_this()](result_t data) {
			f(std::move(data), std::move(self));
		},
		a_flnk_populate(), std::move(child_type_id), true
	);
}

auto fusion_link::bridge() const -> sp_fusion {
	return actorf<sp_fusion>(factor_, a_flnk_bridge()).value_or(nullptr);
}

auto fusion_link::reset_bridge(sp_fusion new_bridge) -> void {
	caf::anon_send(actor(), a_flnk_bridge(), std::move(new_bridge));
}

auto fusion_link::propagate_handle() -> result_or_err<sp_node> {
	// set handle of cached node object to this link instance
	auto D = cache();
	self_handle_node(pimpl()->data_);
	return D ? result_or_err<sp_node>(std::move(D)) : tl::make_unexpected(Error::EmptyData);
}

auto fusion_link::cache() const -> sp_node {
	return actorf<sp_node>(factor_, a_lnk_dcache()).value_or(nullptr);
}

NAMESPACE_END(blue_sky::tree)
