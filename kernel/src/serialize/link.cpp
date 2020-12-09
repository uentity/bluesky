/// @file
/// @author uentity
/// @date 28.06.2018
/// @brief BS tree links serialization code
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/serialize/serialize.h>
#include <bs/serialize/tree.h>
#include <bs/serialize/boost_uuid.h>

#include "tree_impl.h"
#include "../tree/link_actor.h"
#include "../tree/hard_link.h"
#include "../tree/fusion_link_actor.h"
#include "../tree/map_engine.h"

#include <cereal/types/chrono.hpp>

using namespace cereal;
using namespace blue_sky;

/*-----------------------------------------------------------------------------
 *  inode
 *-----------------------------------------------------------------------------*/
BSS_FCN_BEGIN(save, tree::inode)
	ar(
		make_nvp("flags", t.flags),
		make_nvp("u", t.u),
		make_nvp("g", t.g),
		make_nvp("o", t.o),
		make_nvp("mod_time", t.mod_time),
		make_nvp("owner", t.owner),
		make_nvp("group", t.group)
	);
BSS_FCN_END

BSS_FCN_BEGIN(load, tree::inode)
	tree::inode::Flags flags;
	ar(make_nvp("flags", flags));
	t.flags = flags;
	std::uint8_t r;
	ar(make_nvp("u", r));
	t.u = r;
	ar(make_nvp("g", r));
	t.g = r;
	ar(make_nvp("o", r));
	t.o = r;
	ar(
		make_nvp("mod_time", t.mod_time),
		make_nvp("owner", t.owner),
		make_nvp("group", t.group)
	);
BSS_FCN_END

BSS_FCN_EXPORT(save, tree::inode)
BSS_FCN_EXPORT(load, tree::inode)

/*-----------------------------------------------------------------------------
 *  link_impl
 *-----------------------------------------------------------------------------*/
BSS_FCN_BEGIN(serialize, tree::link_impl)
	// [NOTE] intentionally do net serialize owner, it will be set up when parent node is loaded
	ar(
		make_nvp("id", t.id_),
		make_nvp("name", t.name_),
		make_nvp("flags", t.flags_)
	);
BSS_FCN_END

CEREAL_REGISTER_POLYMORPHIC_RELATION(tree::engine::impl, tree::link_impl)
CEREAL_REGISTER_TYPE_WITH_NAME(tree::link_impl, "link")
BSS_FCN_EXPORT(serialize, tree::link_impl)

///////////////////////////////////////////////////////////////////////////////
//  ilink_impl
//
BSS_FCN_INL_BEGIN(serialize, tree::ilink_impl)
	// serialize inode
	// [NOTE] disabled - inodes aren't actually used now
	// [TODO] reenable later when conception will be more thought out
	//ar(make_nvp("inode", t.inode_));

	serialize<tree::link_impl>::go(ar, t, version);
BSS_FCN_INL_END(serialize, tree::ilink_impl)

CEREAL_REGISTER_POLYMORPHIC_RELATION(tree::link_impl, tree::ilink_impl)

///////////////////////////////////////////////////////////////////////////////
//  hard_link_impl
//
BSS_FCN_INL_BEGIN(serialize, tree::hard_link_impl)
	if constexpr(Archive::is_saving::value) {
		ar( make_nvp("data", t.data_) );
	}
	else {
		// load data with deferred 2nd trial
		ar(defer_failed(
			t.data_,
			[&t](auto obj) { t.set_data(std::move(obj)); },
			PtrInitTrigger::SuccessAndRetry
		));
	}

	serialize<tree::ilink_impl>::go(ar, t, version);
	//ar( make_nvp("linkbase", base_class<tree::ilink_impl>(&t)) );
BSS_FCN_INL_END(serialize, tree::hard_link_impl)

CEREAL_REGISTER_POLYMORPHIC_RELATION(tree::ilink_impl, tree::hard_link_impl)
CEREAL_REGISTER_TYPE_WITH_NAME(tree::hard_link_impl, "hard_link")

///////////////////////////////////////////////////////////////////////////////
//  weak_link_impl
//
BSS_FCN_INL_BEGIN(serialize, tree::weak_link_impl)
	if constexpr(Archive::is_saving::value) {
		ar( make_nvp("data", t.data_.lock()) );
	}
	else {
		auto unused = sp_obj{};
		ar(defer_failed(
			unused,
			[&t](auto obj) { t.set_data(std::move(obj)); },
			PtrInitTrigger::SuccessAndRetry
		));
	}

	serialize<tree::ilink_impl>::go(ar, t, version);
	//ar( make_nvp("linkbase", base_class<tree::ilink_impl>(&t)) );
BSS_FCN_INL_END(serialize, tree::weak_link_impl)

CEREAL_REGISTER_POLYMORPHIC_RELATION(tree::ilink_impl, tree::weak_link_impl)
CEREAL_REGISTER_TYPE_WITH_NAME(tree::weak_link_impl, "weak_link")

///////////////////////////////////////////////////////////////////////////////
//  sym_link_impl
//
BSS_FCN_INL_BEGIN(serialize, tree::sym_link_impl)
	ar(make_nvp("path", t.path_));

	serialize<tree::link_impl>::go(ar, t, version);
BSS_FCN_INL_END(serialize, tree::sym_link_impl)

CEREAL_REGISTER_POLYMORPHIC_RELATION(tree::link_impl, tree::sym_link_impl)
CEREAL_REGISTER_TYPE_WITH_NAME(tree::sym_link_impl, "sym_link")

///////////////////////////////////////////////////////////////////////////////
//  fusion_link_impl
//
BSS_FCN_INL_BEGIN(serialize, tree::fusion_link_impl)
	using namespace blue_sky::tree;
	if constexpr(Archive::is_saving::value) {
		ar( make_nvp("bridge", t.bridge_) );
		// save status values
		ar(
			make_nvp("data_status", t.req_status(Req::Data)),
			make_nvp("dnode_status", t.req_status(Req::DataNode))
		);
		// save cached object
		ar( make_nvp("data", t.data_) );
	}
	else {
		// load bridge with deferred trial
		ar(defer_failed(
			t.bridge_,
			[&t](auto B) { t.bridge_ = std::move(B); },
			PtrInitTrigger::SuccessAndRetry
		));
		// restore status
		tree::ReqStatus s;
		ar(make_nvp("data_status", s));
		t.rs_reset(Req::Data, ReqReset::Always, s);
		ar(make_nvp("dnode_status", s));
		t.rs_reset(Req::DataNode, ReqReset::Always, s);

		// load data with deferred 2nd trial
		ar(defer_failed(
			t.data_,
			[&](auto obj) { t.data_ = std::move(obj); },
			PtrInitTrigger::SuccessAndRetry
		));
	}

	serialize<tree::ilink_impl>::go(ar, t, version);
	//ar( make_nvp("linkbase", base_class<tree::ilink_impl>(&t)) );
BSS_FCN_INL_END(serialize, tree::fusion_link_impl)

CEREAL_REGISTER_POLYMORPHIC_RELATION(tree::ilink_impl, tree::fusion_link_impl)
CEREAL_REGISTER_TYPE_WITH_NAME(tree::fusion_link_impl, "fusion_link")

///////////////////////////////////////////////////////////////////////////////
//  map_link - not serialized
//
CEREAL_REGISTER_POLYMORPHIC_RELATION(tree::link_impl, tree::map_link_impl)
CEREAL_REGISTER_TYPE_WITH_NAME(tree::map_link_impl, "map_link")

/*-----------------------------------------------------------------------------
 *  link
 *-----------------------------------------------------------------------------*/
BSS_FCN_BEGIN(serialize, tree::link)
	// [NOTE] intentionally do net serialize owner, it will be set up when parent node is loaded
	if constexpr(Archive::is_saving::value) {
		// for nil & map links save empty impl pointer
		const auto Limpl = !t || t.type_id() == tree::map_link::type_id_() ?
			tree::sp_limpl{} : t.pimpl_;
		ar(make_nvp("link", Limpl));
	}
	else {
		ar(make_nvp("link", t.pimpl_));
		if(!t.pimpl_) t = tree::link{};
		// ID is ready, we can start internal actor
		t.start_engine();
		// if pointee is a node, correct it's handle
		t.pimpl()->propagate_handle();
	}
BSS_FCN_END

BSS_FCN_EXPORT(serialize, tree::link)

BSS_REGISTER_DYNAMIC_INIT(link)
