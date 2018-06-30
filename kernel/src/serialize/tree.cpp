/// @file
/// @author uentity
/// @date 22.06.2018
/// @brief Implementation of BS tree-related serialization
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/serialize/tree.h>
#include <bs/serialize/boost_uuid.h>
#include <bs/serialize/base_types.h>
#include <cereal/types/vector.hpp>

#include "../node_impl.h"

//CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(
//	blue_sky::tree::node::links_container, cereal::specialization::non_member_load_save
//)

NAMESPACE_BEGIN(blue_sky)
using namespace cereal;
using namespace tree;

///*-----------------------------------------------------------------------------
// *  multi_index
// *-----------------------------------------------------------------------------*/
//BSS_FCN_INL_BEGIN(save, node::links_container)
//	//t.save(ar, version);
//	//boost::serialization::serialize(ar, t, version);
//BSS_FCN_INL_END
//
//BSS_FCN_INL_BEGIN(load, node::links_container)
//	//t.load(ar, version);
//	//boost::serialization::serialize(ar, t, version);
//BSS_FCN_INL_END

/*-----------------------------------------------------------------------------
 *  node::node_impl
 *-----------------------------------------------------------------------------*/
BSS_FCN_INL_BEGIN(save, node::node_impl)
	ar(
		make_nvp("handle", t.handle_),
		make_nvp("allowed_otypes", t.allowed_otypes_)
	);
	// save leafs
	ar(make_size_tag(t.links_.size()));
	for(const auto& leaf : t.links_)
		ar(leaf);
BSS_FCN_INL_END(save, node::node_impl)

BSS_FCN_INL_BEGIN(load, node::node_impl)
	ar(
		make_nvp("handle", t.handle_),
		make_nvp("allowed_otypes", t.allowed_otypes_)
	);
	// load leafs
	std::size_t sz;
	ar(make_size_tag(sz));
	for(std::size_t i = 0; i < sz; ++i) {
		sp_link leaf;
		ar(leaf);
		t.insert(std::move(leaf), node::InsertPolicy::AllowDupNames);
	}
BSS_FCN_INL_END(load, node::node_impl)

/*-----------------------------------------------------------------------------
 *  node
 *-----------------------------------------------------------------------------*/
BSS_FCN_BEGIN(serialize, node)
	ar(
		make_nvp("objbase", base_class<objbase>(&t)),
		make_nvp("impl", t.pimpl_)
	);
	// correct owner of all loaded links
	t.propagate_owner();
	//node::links_container tmp;
	//ar(tmp);
BSS_FCN_END

BSS_FCN_EXPORT(serialize, node)

NAMESPACE_END(blue_sky)

BSS_REGISTER_TYPE(blue_sky::tree::node)

BSS_REGISTER_DYNAMIC_INIT(node)

