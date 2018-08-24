/// @file
/// @author uentity
/// @date 28.06.2018
/// @brief BS tree links serialization code
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/serialize/tree.h>
#include <bs/serialize/boost_uuid.h>
#include <bs/serialize/base_types.h>
#include "../tree/link_impl.h"
#include "../tree/fusion_link_impl.h"

#include <cereal/types/polymorphic.hpp>

using namespace cereal;
using namespace blue_sky;

/*-----------------------------------------------------------------------------
 *  inode
 *-----------------------------------------------------------------------------*/
BSS_FCN_BEGIN(save, tree::inode)
	ar(
		make_nvp("owner", t.owner),
		make_nvp("group", t.group),
		make_nvp("suid", t.suid),
		make_nvp("sgid", t.sgid),
		make_nvp("sticky", t.sticky),
		make_nvp("u", t.u),
		make_nvp("g", t.g),
		make_nvp("o", t.o)
	);
BSS_FCN_END

BSS_FCN_BEGIN(load, tree::inode)
	ar(
		make_nvp("owner", t.owner),
		make_nvp("group", t.group)
	);
	bool bit;
	ar(make_nvp("suid", bit));
	t.suid = bit;
	ar(make_nvp("sgid", bit));
	t.sgid = bit;
	ar(make_nvp("sticky", bit));
	t.sticky = bit;

	std::uint8_t flags;
	ar(make_nvp("u", flags));
	t.u = flags;
	ar(make_nvp("g", flags));
	t.g = flags;
	ar(make_nvp("o", flags));
	t.o = flags;
BSS_FCN_END

BSS_FCN_EXPORT(save, tree::inode)
BSS_FCN_EXPORT(load, tree::inode)

/*-----------------------------------------------------------------------------
 *  link
 *-----------------------------------------------------------------------------*/
BSS_FCN_BEGIN(serialize, tree::link)
	// [NOTE] intentionally DON'T save name,
	// because name will be saved in derived classes to invoke minimal constructor
	// also do not save owner, because owner will be correctly set by `node`
	ar(
//		make_nvp("name", t.name_),
		make_nvp("id", t.pimpl_->id_),
		make_nvp("flags", t.pimpl_->flags_),
		make_nvp("inode", t.pimpl_->inode_)
		// intentionally do net serialize owner, it will be set up when parent node is loaded
	);
BSS_FCN_END

BSS_FCN_EXPORT(serialize, tree::link)

/*-----------------------------------------------------------------------------
 *  hard_link
 *-----------------------------------------------------------------------------*/
// provide non-empty constructor
BSS_FCN_BEGIN(load_and_construct, tree::hard_link)
	// load name? data & construct instance
	std::string name;
	sp_obj data;
	ar(name, data);
	construct(std::move(name), std::move(data));
	// load base link
	ar( make_nvp("link_base", base_class<tree::link>(construct.ptr())) );
BSS_FCN_END

BSS_FCN_BEGIN(serialize, tree::hard_link)
	ar(
		make_nvp("name", t.pimpl_->name_),
		make_nvp("data", t.data_),
		make_nvp("link_base", base_class<tree::link>(&t))
	);
BSS_FCN_END

BSS_FCN_EXPORT(serialize, tree::hard_link)
BSS_FCN_EXPORT(load_and_construct, tree::hard_link)

/*-----------------------------------------------------------------------------
 *  weak_link
 *-----------------------------------------------------------------------------*/
// provide non-empty constructor
BSS_FCN_BEGIN(load_and_construct, tree::weak_link)
	// load name? data & construct instance
	std::string name;
	sp_obj data;
	ar(name, data);
	construct(std::move(name), std::move(data));
	// load base link
	ar( make_nvp("link_base", base_class<tree::link>(construct.ptr())) );
BSS_FCN_END

BSS_FCN_BEGIN(serialize, tree::weak_link)
	ar(
		make_nvp("name", t.pimpl_->name_),
		make_nvp("data", t.data_.lock()),
		make_nvp("link_base", base_class<tree::link>(&t))
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
	// load base link
	ar( make_nvp("link_base", base_class<tree::link>(construct.ptr())) );
BSS_FCN_END

BSS_FCN_BEGIN(serialize, tree::sym_link)
	ar(
		make_nvp("name", t.pimpl_->name_),
		make_nvp("path", t.path_),
		make_nvp("link_base", base_class<tree::link>(&t))
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
		make_nvp("data", t.pimpl_->data_),
		make_nvp("pop_status", t.pimpl_->pop_status_),
		make_nvp("data_status", t.pimpl_->data_status_),
		make_nvp("link_base", base_class<tree::link>(&t))
	);
BSS_FCN_END

BSS_FCN_BEGIN(load_and_construct, tree::fusion_link)
	// load base data & construct instance
	std::string name;
	tree::sp_node data;
	tree::sp_fusion bridge;
	ar(name, bridge, data);
	construct(std::move(name), std::move(bridge), std::move(data));
	// load other data
	auto& t = *construct.ptr();
	ar(
		make_nvp("pop_status", t.pimpl_->pop_status_),
		make_nvp("data_status", t.pimpl_->data_status_),
		// base link
		make_nvp("link_base", base_class<tree::link>(&t))
	);
BSS_FCN_END

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

