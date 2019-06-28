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

#include "../tree/link_impl.h"
#include "../tree/fusion_link_impl.h"

#include <cereal/types/chrono.hpp>

using namespace cereal;
using namespace blue_sky;
using Req = blue_sky::tree::link::Req;
using ReqStatus = blue_sky::tree::link::ReqStatus;

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
 *  link
 *-----------------------------------------------------------------------------*/
BSS_FCN_BEGIN(serialize, tree::link)
	// [NOTE] intentionally DON'T save name,
	// because name will be saved in derived classes to invoke minimal constructor
	// [NOTE] intentionally do net serialize owner, it will be set up when parent node is loaded
	ar(
		make_nvp("id", t.pimpl_->id_),
		make_nvp("flags", t.pimpl_->flags_)
	);
	// assume link is root by default -- it's safe when restoring link or tree from archive
	if constexpr(Archive::is_loading::value)
		t.propagate_handle();
BSS_FCN_END

BSS_FCN_EXPORT(serialize, tree::link)

/*-----------------------------------------------------------------------------
 *  ilink
 *-----------------------------------------------------------------------------*/
BSS_FCN_BEGIN(serialize, tree::ilink)
	// serialize inode
	ar(make_nvp("inode", t.inode_));
	// serialize base link by direct call in order to omit creating inner JSON section
	serialize<tree::link>::go(ar, t, version);
BSS_FCN_END

CEREAL_REGISTER_POLYMORPHIC_RELATION(tree::link, tree::ilink)
BSS_FCN_EXPORT(serialize, tree::ilink)

/*-----------------------------------------------------------------------------
 *  hard_link
 *-----------------------------------------------------------------------------*/
// provide non-empty constructor
BSS_FCN_BEGIN(load_and_construct, tree::hard_link)
	// load name & construct instance
	std::string name;
	ar(name);
	construct(std::move(name), sp_obj{});
	auto plnk = construct.ptr();

	// helper that sets OK status on successfull object deserialization
	static const auto update_status = [](auto& L) {
		L.data_impl().map([&L](const sp_obj& obj) {
			if(!obj) return;
			L.rs_reset(Req::Data, ReqStatus::OK);
			if(obj->is_node())
				L.rs_reset(Req::DataNode, ReqStatus::OK);
		});
	};
	// load data with deferred 2nd trial
	ar( defer_failed(plnk->data_, [plnk](sp_obj obj) {
		plnk->data_ = std::move(obj);
		update_status(*plnk);
	}) );
	// if 1st trial succeeded, update status
	update_status(*plnk);

	// load base link
	ar( base_class<tree::ilink>(plnk) );
BSS_FCN_END

BSS_FCN_BEGIN(serialize, tree::hard_link)
	ar(
		make_nvp("name", t.pimpl_->name_),
		make_nvp("data", t.data_),
		make_nvp("linkbase", base_class<tree::ilink>(&t))
	);
BSS_FCN_END

BSS_FCN_EXPORT(serialize, tree::hard_link)
BSS_FCN_EXPORT(load_and_construct, tree::hard_link)

/*-----------------------------------------------------------------------------
 *  weak_link
 *-----------------------------------------------------------------------------*/
// provide non-empty constructor
BSS_FCN_BEGIN(load_and_construct, tree::weak_link)
	// load name & construct instance
	std::string name;
	ar(name);
	sp_obj data;
	construct(std::move(name), data);
	auto plnk = construct.ptr();

	// helper that sets OK status on successfull object deserialization
	static const auto update_status = [](auto& L) {
		L.data_impl().map([&L](const sp_obj& obj) {
			if(!obj) return;
			L.rs_reset(Req::Data, ReqStatus::OK);
			if(obj->is_node())
				L.rs_reset(Req::DataNode, ReqStatus::OK);
		});
	};
	// load data with deferred 2nd trial
	ar( defer_failed(data, [plnk](const sp_obj& obj) {
		plnk->data_ = obj;
		update_status(*plnk);
	}) );
	// if 1st trial succeeded, update status
	if(data) plnk->data_ = data;
	update_status(*plnk);

	// load base link
	ar( base_class<tree::ilink>(plnk) );
BSS_FCN_END

BSS_FCN_BEGIN(serialize, tree::weak_link)
	ar(
		make_nvp("name", t.pimpl_->name_),
		make_nvp("data", t.data_.lock()),
		make_nvp("linkbase", base_class<tree::ilink>(&t))
	);
BSS_FCN_END

BSS_FCN_EXPORT(serialize, tree::weak_link)
BSS_FCN_EXPORT(load_and_construct, tree::weak_link)

/*-----------------------------------------------------------------------------
 *  sym_link
 *-----------------------------------------------------------------------------*/
// provide non-empty constructor
BSS_FCN_BEGIN(load_and_construct, tree::sym_link)
	// load both name and path & construct instance
	std::string name, path;
	ar(name, path);
	construct(std::move(name), std::move(path));
	auto plnk = construct.ptr();

	// load base link
	ar( base_class<tree::link>(plnk) );
	// defer update symlink Data status
	ar( defer(Functor{ [plnk](auto&) { plnk->check_alive(); } }) );
BSS_FCN_END

BSS_FCN_BEGIN(serialize, tree::sym_link)
	ar(
		make_nvp("name", t.pimpl_->name_),
		make_nvp("path", t.path_),
		make_nvp("linkbase", base_class<tree::link>(&t))
	);
BSS_FCN_END

BSS_FCN_EXPORT(serialize, tree::sym_link)
BSS_FCN_EXPORT(load_and_construct, tree::sym_link)

/*-----------------------------------------------------------------------------
 *  fusion_link
 *-----------------------------------------------------------------------------*/
BSS_FCN_BEGIN(serialize, tree::fusion_link)
	ar(
		make_nvp("name", static_cast<tree::link&>(t).pimpl_->name_),
		make_nvp("bridge", t.pimpl_->bridge_),
		make_nvp("linkbase", base_class<tree::ilink>(&t))
	);
BSS_FCN_END

BSS_FCN_BEGIN(load_and_construct, tree::fusion_link)
	// load name & construct instance
	std::string name;
	ar(name);
	construct(std::move(name), tree::sp_node{}, tree::sp_fusion{});
	auto plnk = construct.ptr();

	// load bridge with deferred trial
	ar( defer_failed(plnk->pimpl_->bridge_) );

	// base link
	ar( base_class<tree::ilink>(plnk) );
BSS_FCN_END

BSS_FCN_EXPORT(serialize, tree::fusion_link)
BSS_FCN_EXPORT(load_and_construct, tree::fusion_link)

/*-----------------------------------------------------------------------------
 *  instantiate code for polymorphic types
 *-----------------------------------------------------------------------------*/
using namespace blue_sky;
//CEREAL_REGISTER_TYPE_WITH_NAME(tree::link, "link")
CEREAL_REGISTER_TYPE_WITH_NAME(tree::hard_link, "hard_link")
CEREAL_REGISTER_TYPE_WITH_NAME(tree::weak_link, "weak_link")
CEREAL_REGISTER_TYPE_WITH_NAME(tree::sym_link, "sym_link")
CEREAL_REGISTER_TYPE_WITH_NAME(tree::fusion_link, "fusion_link")

BSS_REGISTER_DYNAMIC_INIT(link)

