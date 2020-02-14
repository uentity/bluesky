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

auto fusion_link_impl::spawn_actor(std::shared_ptr<link_impl> limpl) const -> caf::actor {
	return spawn_lactor<fusion_link_actor>(std::move(limpl));
}

auto fusion_link_impl::clone(bool deep) const -> sp_limpl {
	return std::make_shared<fusion_link_impl>(
		name_,
		deep ? kernel::tfactory::clone_object(data_) : data_,
		bridge_, flags_
	);
}
auto fusion_link_impl::propagate_handle(const link& L) -> result_or_err<sp_node> {
	// set handle of cached node object to this link instance
	set_node_handle(L, data_);
	return data_ ? result_or_err<sp_node>(data_) : tl::make_unexpected(Error::EmptyData);
}

LIMPL_TYPE_DEF(fusion_link_impl, "fusion_link")

/*-----------------------------------------------------------------------------
 *  fusion_link
 *-----------------------------------------------------------------------------*/
#define FIMPL static_cast<fusion_link_impl&>(*pimpl())

fusion_link::fusion_link(
	std::string name, sp_node data, sp_fusion bridge, Flags f
) : // set LazyLoad flag by default
	super(std::make_shared<fusion_link_impl>( std::move(name), data, bridge, Flags(f | LazyLoad) ))
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
		bserr() << log::E("fusion_link: cannot create object of type '{}'! Empty link!") <<
			obj_type << log::end;
}

fusion_link::fusion_link()
	: super(std::make_shared<fusion_link_impl>(), false)
{}

LINK_CONVERT_TO(fusion_link)

auto fusion_link::populate(const std::string& child_type_id, bool wait_if_busy) const
-> result_or_err<sp_node> {
	return pimpl()->actorf<result_or_errbox<sp_node>>(
		*this, a_flnk_populate(), child_type_id, wait_if_busy
	);
}

auto fusion_link::populate(link::process_data_cb f, std::string child_type_id) const -> void {
	using result_t = result_or_errbox<sp_node>;

	anon_request(
		actor(*this), def_timeout(true), false,
		[f = std::move(f), self = *this](result_t data) mutable {
			f( std::move(data), std::move(self) );
		},
		a_flnk_populate(), std::move(child_type_id), true
	);
}

auto fusion_link::bridge() const -> sp_fusion {
	return pimpl()->actorf<result_or_errbox<sp_fusion>>(
		*this, a_flnk_bridge()
	).value_or(nullptr);
}

auto fusion_link::reset_bridge(sp_fusion new_bridge) -> void {
	caf::anon_send(actor(*this), a_flnk_bridge(), std::move(new_bridge));
}

auto fusion_link::cache() const -> sp_node {
	return std::static_pointer_cast<tree::node>(
		pimpl()->actorf<sp_obj>(*this, a_lnk_dcache()).value_or(nullptr)
	);
}

auto fusion_link::type_id_() -> std::string_view {
	return fusion_link_impl::type_id_();
}

NAMESPACE_END(blue_sky::tree)
