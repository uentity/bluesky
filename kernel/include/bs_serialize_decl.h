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

namespace detail {

template< class T >
struct bs_init_eti;

}

/// @brief Force boost::serialization::extended_type_info creation (and registering)
///
/// @tparam T
template< class T >
void serialize_register_eti();

/*-----------------------------------------------------------------
 * check whether given serialization data fix is applicable to given type
 *----------------------------------------------------------------*/
template< class T, class fixer >
struct serialize_fix_applicable {
	// does given fixer applicable to given type T during saving?
	typedef boost::false_type on_save;
	// and during loading
	typedef boost::false_type on_load;
	// type returned by fixer save
	typedef T save_ret_t;
};

}  // eof blue_sky

#endif /* end of include guard: BS_SERIALIZE_DECL_AN05AGKZ */

