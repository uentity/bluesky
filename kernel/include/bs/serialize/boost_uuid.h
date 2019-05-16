/// @file
/// @author uentity
/// @date 28.06.2018
/// @brief Serialization support of boost::uuids::uuid
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "serialize.h"
#include <cereal/types/common.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/string_generator.hpp>

namespace blue_sky {

// save path
template<>
struct atomizer::save<boost::uuids::uuid> {
	using uuid_t = boost::uuids::uuid;
	// check if archive supports binary serialization
	template<typename Archive> using support_binary =
		cereal::traits::is_input_serializable<cereal::BinaryData<uuid_t>, Archive>;
	template<typename Archive> using is_text = cereal::traits::is_text_archive<Archive>;

	// for text archives print text repr of UUID
	template<typename Archive>
	static auto go(Archive& ar, uuid_t const& t) -> std::enable_if_t<is_text<Archive>::value> {
		ar(cereal::make_nvp("textid", boost::uuids::to_string(t)));
	}

	// for all other archives output as array
	template<typename Archive>
	static auto go(Archive& ar, uuid_t const& t) -> std::enable_if_t<!is_text<Archive>::value> {
		cereal::common_detail::serializeArray(ar, t.data, support_binary<Archive>());
	}
};

// load path
template<>
struct atomizer::load<boost::uuids::uuid> {
	using uuid_t = boost::uuids::uuid;
	// check if archive supports binary serialization
	template<typename Archive> using support_binary =
		cereal::traits::is_output_serializable<cereal::BinaryData<uuid_t>, Archive>;
	template<typename Archive> using is_text = cereal::traits::is_text_archive<Archive>;

	// for text archives load text repr of UUID
	template<typename Archive>
	static auto go(Archive& ar, uuid_t& t) -> std::enable_if_t<is_text<Archive>::value> {
		static boost::uuids::string_generator uuid_from_str;

		std::string text_uuid;
		ar(cereal::make_nvp("textid", text_uuid));
		t = uuid_from_str(text_uuid);
	}

	template<typename Archive>
	static auto go(Archive& ar, uuid_t& t) -> std::enable_if_t<!is_text<Archive>::value> {
		cereal::common_detail::serializeArray(ar, t.data, support_binary<Archive>());
	}
};

} /* namespace blue_sky */

