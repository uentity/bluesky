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
	// [NOTE] intentinaly DON'T save name,
	// because name will be saved in derived classes to invoke minimal constructor
	// also do not save owner, because owner will be correctly set by `node`
	ar(
//		make_nvp("name", t.name_),
		make_nvp("id", t.id_),
		make_nvp("flags", t.flags_),
		make_nvp("inode", t.inode_)
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
	ar(base_class<tree::link>(construct.ptr()));
BSS_FCN_END

BSS_FCN_BEGIN(serialize, tree::hard_link)
	ar(
		make_nvp("name", t.name_),
		make_nvp("data", t.data_),
		make_nvp("link_base", base_class<tree::link>(&t))
	);
BSS_FCN_END

BSS_FCN_EXPORT(serialize, tree::hard_link)

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
	ar(base_class<tree::link>(construct.ptr()));
BSS_FCN_END

BSS_FCN_BEGIN(serialize, tree::weak_link)
	ar(
		make_nvp("name", t.name_),
		make_nvp("data", t.data_),
		make_nvp("link_base", base_class<tree::link>(&t))
	);
BSS_FCN_END

BSS_FCN_EXPORT(serialize, tree::weak_link)

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
	ar(base_class<tree::link>(construct.ptr()));
BSS_FCN_END

BSS_FCN_BEGIN(serialize, tree::sym_link)
	ar(
		make_nvp("name", t.name_),
		make_nvp("path", t.path_),
		make_nvp("link_base", base_class<tree::link>(&t))
	);
BSS_FCN_END

BSS_FCN_EXPORT(serialize, tree::sym_link)

// instantiate code for polymorphic types
using namespace blue_sky;
//CEREAL_REGISTER_TYPE_WITH_NAME(tree::link, "link")
CEREAL_REGISTER_TYPE_WITH_NAME(tree::hard_link, "hard_link")
CEREAL_REGISTER_TYPE_WITH_NAME(tree::weak_link, "weak_link")
CEREAL_REGISTER_TYPE_WITH_NAME(tree::sym_link, "sym_link")

BSS_REGISTER_DYNAMIC_INIT(link)

