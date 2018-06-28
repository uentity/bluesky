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

NAMESPACE_BEGIN(blue_sky)
using namespace cereal;

/*-----------------------------------------------------------------------------
 *  inode
 *-----------------------------------------------------------------------------*/
BSS_FCN_BEGIN(save, tree::inode)
	ar(
		make_nvp("owner", t.owner),
		make_nvp("group", t.group)
	);
	bool bit = t.suid;
	ar(make_nvp("suid", bit));
	bit = t.sgid;
	ar(make_nvp("sgid", bit));
	bit = t.sticky;
	ar(make_nvp("sticky", bit));
	bit = t.u;
	ar(make_nvp("u", bit));
	bit = t.g;
	ar(make_nvp("g", bit));
	bit = t.o;
	ar(make_nvp("o", bit));
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
	ar(make_nvp("u", bit));
	t.u = bit;
	ar(make_nvp("g", bit));
	t.g = bit;
	ar(make_nvp("o", bit));
	t.o = bit;
BSS_FCN_END

BSS_FCN_EXPORT(save, tree::inode)
BSS_FCN_EXPORT(load, tree::inode)

/*-----------------------------------------------------------------------------
 *  link
 *-----------------------------------------------------------------------------*/
BSS_FCN_BEGIN(serialize, tree::link)
	ar(
		make_nvp("name", t.name_),
		make_nvp("id", t.id_),
		make_nvp("flags", t.flags_),
		make_nvp("inode", t.inode_)
		//make_nvp("owner", t.owner_)
	);
BSS_FCN_END

BSS_FCN_EXPORT(serialize, tree::link)

/*-----------------------------------------------------------------------------
 *  hard_link
 *-----------------------------------------------------------------------------*/
BSS_FCN_BEGIN(serialize, tree::hard_link)
	ar(
		make_nvp("link_base", base_class<tree::link>(&t)),
		make_nvp("data", t.data_)
	);
BSS_FCN_END

BSS_FCN_EXPORT(serialize, tree::hard_link)

/*-----------------------------------------------------------------------------
 *  weak_link
 *-----------------------------------------------------------------------------*/
BSS_FCN_BEGIN(serialize, tree::weak_link)
	ar(
		make_nvp("link_base", base_class<tree::link>(&t)),
		make_nvp("data", t.data_)
	);
BSS_FCN_END

BSS_FCN_EXPORT(serialize, tree::weak_link)

/*-----------------------------------------------------------------------------
 *  sym_link
 *-----------------------------------------------------------------------------*/
BSS_FCN_BEGIN(serialize, tree::sym_link)
	ar(
		make_nvp("link_base", base_class<tree::link>(&t)),
		make_nvp("path", t.path_)
	);
BSS_FCN_END

BSS_FCN_EXPORT(serialize, tree::sym_link)

NAMESPACE_END(blue_sky)

BSS_REGISTER_DYNAMIC_INIT(link)

