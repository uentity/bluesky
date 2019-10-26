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

#include "../tree/link_actor.h"
#include "../tree/fusion_link_actor.h"

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
	// [NOTE] intentionally do net serialize owner, it will be set up when parent node is loaded
	ar(
		make_nvp("id", t.pimpl_->id_),
		make_nvp("name", t.pimpl_->name_),
		make_nvp("flags", t.pimpl_->flags_)
	);

	if constexpr(Archive::is_loading::value) {
		// ID is ready, we can start internal actor
		t.start_engine();
		// assume link is root by default -- it's safe when restoring link or tree from archive
		t.propagate_handle();
	}
BSS_FCN_END

BSS_FCN_EXPORT(serialize, tree::link)

/*-----------------------------------------------------------------------------
 *  ilink
 *-----------------------------------------------------------------------------*/
BSS_FCN_BEGIN(serialize, tree::ilink)
	// serialize inode
	ar(make_nvp("inode", t.pimpl()->inode_));
	// serialize base link by direct call in order to omit creating inner JSON section
	serialize<tree::link>::go(ar, t, version);
BSS_FCN_END

CEREAL_REGISTER_POLYMORPHIC_RELATION(tree::link, tree::ilink)
BSS_FCN_EXPORT(serialize, tree::ilink)

/*-----------------------------------------------------------------------------
 *  hard_link
 *-----------------------------------------------------------------------------*/
#define SERIALIZE_LINK_WDATA                                         \
    if constexpr(!Archive::is_loading::value) {                      \
        auto guard = std::shared_lock{ t.pimpl()->guard_ };          \
        ar( make_nvp("data", t.pimpl()->data_) );                    \
    }                                                                \
    else {                                                           \
        ar(defer_failed(                                             \
            t.pimpl()->data_,                                        \
            [&t](auto obj) { t.pimpl()->set_data(std::move(obj)); }, \
            PtrInitTrigger::SuccessAndRetry                          \
        ));                                                          \
    }                                                                \
    ar( make_nvp("linkbase", base_class<tree::ilink>(&t)) );

// don't lock when loading - object is not yet 'usable' and fully constructed
#define GUARD                                       \
[[maybe_unused]] auto guard = [&] {                 \
    if constexpr(Archive::is_saving::value)         \
        return std::shared_lock{t.pimpl()->guard_}; \
    else                                            \
        return 0;                                   \
}();

BSS_FCN_BEGIN(serialize, tree::hard_link)
    GUARD
	ar( make_nvp("linkbase", base_class<tree::ilink>(&t)) );

	if constexpr(!Archive::is_loading::value) {
		ar( make_nvp("data", t.pimpl()->data_) );
	}
	else {
		// load data with deferred 2nd trial
		ar(defer_failed(
			t.pimpl()->data_,
			[&t](auto obj) { t.pimpl()->set_data(std::move(obj)); },
			PtrInitTrigger::SuccessAndRetry
		));
	}
BSS_FCN_END

BSS_FCN_EXPORT(serialize, tree::hard_link)

/*-----------------------------------------------------------------------------
 *  weak_link
 *-----------------------------------------------------------------------------*/
BSS_FCN_BEGIN(serialize, tree::weak_link)
    GUARD
	ar( make_nvp("linkbase", base_class<tree::ilink>(&t)) );

	if constexpr(!Archive::is_loading::value) {
		ar( make_nvp("data", t.pimpl()->data_.lock()) );
	}
	else {
		auto unused = sp_obj{};
		ar(defer_failed(
			unused,
			[&t](auto obj) { t.pimpl()->set_data(std::move(obj)); },
			PtrInitTrigger::SuccessAndRetry
		));
	}
BSS_FCN_END

BSS_FCN_EXPORT(serialize, tree::weak_link)

/*-----------------------------------------------------------------------------
 *  sym_link
 *-----------------------------------------------------------------------------*/
BSS_FCN_BEGIN(serialize, tree::sym_link)
    GUARD
	ar(
		make_nvp("linkbase", base_class<tree::link>(&t)),
		make_nvp("path", t.pimpl()->path_)
	);
BSS_FCN_END

BSS_FCN_EXPORT(serialize, tree::sym_link)

/*-----------------------------------------------------------------------------
 *  fusion_link
 *-----------------------------------------------------------------------------*/
BSS_FCN_BEGIN(serialize, tree::fusion_link)
    GUARD
	ar( make_nvp("linkbase", base_class<tree::ilink>(&t)) );

	if constexpr(!Archive::is_loading::value) {
		ar( make_nvp("bridge", t.pimpl()->bridge_) );
	}
	else {
		// load bridge with deferred trial
		ar(defer_failed(
			t.pimpl()->bridge_,
			[&t](auto B) { t.pimpl()->bridge_ = std::move(B); },
			PtrInitTrigger::SuccessAndRetry
		));
	}
BSS_FCN_END

BSS_FCN_EXPORT(serialize, tree::fusion_link)

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

