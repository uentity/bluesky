/// @file
/// @author uentity
/// @date 22.06.2018
/// @brief Implementation of BS tree-related serialization
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/log.h>
#include "../tree/node_actor.h"
#include <bs/serialize/serialize.h>
#include <bs/serialize/tree.h>

#include <fstream>
#include <cereal/types/vector.hpp>

NAMESPACE_BEGIN(blue_sky)
using namespace cereal;
using namespace tree;
using blue_sky::detail::shared;

/*-----------------------------------------------------------------------------
 *  node::node_impl
 *-----------------------------------------------------------------------------*/
namespace {
// proxy leafs view to serialize 'em as separate block or 'unit'
struct leafs_view {
	node_impl& N;

	leafs_view(const node_impl& N_) : N(const_cast<node_impl&>(N_)) {}

	template<typename Archive>
	auto save(Archive& ar) const -> void {
		ar(make_size_tag(N.links_.size()));
		// save links in custom index order
		const auto& any_order = N.links_.get<Key_tag<Key::AnyOrder>>();
		for(const auto& leaf : any_order)
			ar(leaf);
	}

	template<typename Archive>
	auto load(Archive& ar) -> void {
		std::size_t sz;
		ar(make_size_tag(sz));
		// load links in custom index order
		auto& any_order = N.links_.get<Key_tag<Key::AnyOrder>>();
		for(std::size_t i = 0; i < sz; ++i) {
			sp_link leaf;
			ar(leaf);
			//N.insert(std::move(leaf));
			any_order.insert(any_order.end(), std::move(leaf));
		}
	}
};

} // eof hidden namespace

BSS_FCN_INL_BEGIN(serialize, node_impl)
	[[maybe_unused]] auto guard = [&] {
		if constexpr(Archive::is_saving::value)
			return t.lock(shared);
		else // don't lock when loading - object is not yet 'usable' and fully constructed
			return 0;
	}();

	ar(
		make_nvp("allowed_otypes", t.allowed_otypes_),
		make_nvp("leafs", leafs_view(t))
	);
BSS_FCN_INL_END(save, node_impl)

/*-----------------------------------------------------------------------------
 *  node
 *-----------------------------------------------------------------------------*/
BSS_FCN_BEGIN(serialize, node)
	ar(
		make_nvp("objbase", base_class<objbase>(&t)),
		make_nvp("node_impl", *t.pimpl_)
	);
	// save actor group's ID
	if constexpr(Archive::is_saving::value) {
		const auto& ngrp = t.pimpl_->self_grp;
		ar(make_nvp("gid", ngrp ? ngrp.get()->identifier() : ""));
	}
BSS_FCN_END

BSS_FCN_BEGIN(load_and_construct, node)
	construct(false);
	auto& t = *construct.ptr();
	::cereal::serialize(ar, t, version);
	// start actor (load group ID first)
	std::string gid;
	ar(make_nvp("gid", gid));
	t.start_engine(gid);
	// correct owner of all loaded links
	t.propagate_owner();
BSS_FCN_END

BSS_FCN_EXPORT(serialize, node)
BSS_FCN_EXPORT(load_and_construct, node)

NAMESPACE_BEGIN(tree)
/*-----------------------------------------------------------------------------
 *  tree save/load impl
 *-----------------------------------------------------------------------------*/
auto save_tree(const sp_link& root, const std::string& filename, TreeArchive ar, timespan max_wait) -> error {
	if(ar == TreeArchive::FS) {
		auto ar = tree_fs_output(filename);
		ar(root);
		auto errs = ar.wait_objects_saved(max_wait);
		// collect all errors happened
		std::string reduced_er;
		for(const auto& er : errs) {
			if(er) {
				if(!reduced_er.empty()) reduced_er += '\n';
				reduced_er += er.what();
			}
		}
		return reduced_er.empty() ? success() : error::quiet(reduced_er);
	}
	// open file for writing
	std::ofstream fs(
		filename,
		std::ios::out | std::ios::trunc | (ar == TreeArchive::Binary ? std::ios::binary : std::ios::openmode())
	);
	if(!fs) return error(std::string("Cannot create file {}") + filename);

	// dump link to JSON archive
	if(ar == TreeArchive::Binary) {
		cereal::PortableBinaryOutputArchive ja(fs);
		ja(root);
	}
	else {
		cereal::JSONOutputArchive ja(fs);
		ja(root);
	}
	return success();
}

auto load_tree(const std::string& filename, TreeArchive ar) -> result_or_err<sp_link> {
	sp_link res;
	if(ar == TreeArchive::FS) {
		auto ar = tree_fs_input(filename);
		ar(res);
		ar.serializeDeferments();
		return res;
	}

	// open file for reading
	std::ifstream fs(
		filename,
		std::ios::in | (ar == TreeArchive::Binary ? std::ios::binary : std::ios::openmode())
	);
	if(!fs) return tl::make_unexpected(error(std::string("Cannot create file {}") + filename));

	// load link from JSON archive
	if(ar == TreeArchive::Binary) {
		cereal::PortableBinaryInputArchive ja(fs);
		ja(res);
	}
	else {
		cereal::JSONInputArchive ja(fs);
		ja(res);
	}
	return res;
}

NAMESPACE_END(tree)
NAMESPACE_END(blue_sky)

BSS_REGISTER_TYPE(blue_sky::tree::node)

BSS_REGISTER_DYNAMIC_INIT(node)

