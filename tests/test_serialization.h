/// @file
/// @author uentity
/// @date 11.04.2019
/// @brief Helpers for quick test serialization of any type
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/objbase.h>
#include <bs/log.h>
#include <bs/serialize/serialize.h>

#include <boost/interprocess/streams/vectorstream.hpp>
#include <boost/test/unit_test.hpp>

template<bool Binary>
struct select_archives {
	using InputArchive = cereal::JSONInputArchive;
	using OutputArchive = cereal::JSONOutputArchive;
	using Stream = std::stringstream;
	using StreamData = std::string;

	static auto stream_data(const Stream& S) -> StreamData { return S.str(); }
};
template<>
struct select_archives<true> {
	using InputArchive = cereal::PortableBinaryInputArchive;
	using OutputArchive = cereal::PortableBinaryOutputArchive;
	using Stream = boost::interprocess::basic_vectorstream< std::vector<char> >;
	using StreamData = std::vector<char>;

	static auto stream_data(const Stream& S) -> StreamData { return S.vector(); }
};


template<bool Binary = false, typename T = blue_sky::sp_obj>
auto test_save(const T& obj, bool dump_serialized = true) {
	using namespace blue_sky;
	using namespace blue_sky::log;

	using Traits = select_archives<Binary>;
	using OutputArchive = typename Traits::OutputArchive;
	using Stream = typename Traits::Stream;

	std::string dump;
	Stream S;
	{
		OutputArchive ja(S);
		ja(obj);
		ja.serializeDeferments();
		if constexpr(!Binary) {
			if(dump_serialized) {
				dump = S.str();
				bsout() << I("-- Serialized dump:\n{}", dump) << end;
			}
		}
		else { (void)dump_serialized; }
	}
	return Traits::stream_data(S);
}

template<bool Binary = false, typename T = blue_sky::sp_obj>
auto test_load(typename select_archives<Binary>::StreamData data, T& obj) -> void {
	using namespace blue_sky;
	using Traits = select_archives<Binary>;
	using InputArchive = typename Traits::InputArchive;
	using Stream = typename Traits::Stream;

	Stream S(std::move(data));
	{
		InputArchive ja(S);
		ja(obj);
		ja.serializeDeferments();
	}
}

template<bool Binary = false, typename U = blue_sky::sp_obj, typename V = U>
decltype(auto) test_saveload(const U& src, V& tar, bool dump_serialized = true) {
	test_load<Binary>(test_save<Binary>(src, dump_serialized), tar);
	return tar;
}

template<bool Binary = false, typename U = blue_sky::sp_obj>
auto test_saveload(const U& src, bool dump_serialized = true) {
	U tar;
	test_load<Binary>(test_save<Binary>(src, dump_serialized), tar);
	if constexpr(std::is_convertible_v<U, blue_sky::sp_cobj>)
		BOOST_TEST(src->id() == tar->id());
	return tar;
}

template<typename... Ts>
decltype(auto) test_json(Ts&&... ts) {
	return test_saveload<false>(std::forward<Ts>(ts)...);
}

template<typename... Ts>
decltype(auto) test_binary(Ts&&... ts) {
	return test_saveload<true>(std::forward<Ts>(ts)...);
}

