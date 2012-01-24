// This file is part of BlueSky
// 
// BlueSky is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.
// 
// BlueSky is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with BlueSky; if not, see <http://www.gnu.org/licenses/>.

#ifndef BS_SERIALIZE_DECL_AN05AGKZ
#define BS_SERIALIZE_DECL_AN05AGKZ

#include "setup_plugin_api.h"

namespace blue_sky {

class BS_API_PLUGIN bs_serialize {
public:
	template< class Archive, class T >
	struct save {
		static void go(
			Archive&,
			const T&,
			const unsigned int
		)
		{}
	};

	template< class Archive, class T >
	struct load {
		static void go (
			Archive&,
			T&,
			const unsigned int
		)
		{}
	};

	template< class Archive, class T >
	struct serialize {
		static void go(
			Archive&,
			T&,
			const unsigned int
		)
		{}
	};

	template< class Archive, class T >
	struct save_construct_data {
		static void go (
			Archive&,
			const T*,
			const unsigned int
		)
		{}
	};

	template< class Archive, class T >
	struct load_construct_data {
		static void go (
			Archive&,
			T*,
			const unsigned int
		)
		{}
	};
};

}  // eof blue_sky

// redirect serialization free functions to blue_sky::bs_serialize
//namespace boost { namespace serialization {
//
//template< class Archive, class T >
//BS_API_PLUGIN void serialize(
//	Archive& ar,
//	T& t,
//	const unsigned int version
//){
//	::blue_sky::bs_serialize< T >::serialize(ar, t, version);
//}
//
//template< class Archive, class T >
//BS_API_PLUGIN void save_construct_data(
//	Archive& ar,
//	const T* t,
//	const unsigned int version
//){
//	::blue_sky::bs_serialize< T >::save_construct_data(ar, t, version);
//}
//
//template< class Archive, class T >
//BS_API_PLUGIN void load_construct_data(
//	Archive& ar,
//	T* t,
//	const unsigned int version
//){
//	::blue_sky::bs_serialize< T >::load_construct_data(ar, t, version);
//}
//
//template< class Archive, class T >
//static void save(
//	Archive& ar,
//	const T& t,
//	const unsigned int version
//){
//	::blue_sky::bs_serialize< T >::save(ar, t, version);
//}
//
//template< class Archive, class T >
//static void load(
//	Archive& ar,
//	T& t,
//	const unsigned int version
//){
//	::blue_sky::bs_serialize< T >::load(ar, t, version);
//}
//
//}}  // eof boost::serialization

#endif /* end of include guard: BS_SERIALIZE_DECL_AN05AGKZ */

