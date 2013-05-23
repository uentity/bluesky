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

#ifndef BS_SERIALIZE_FIXDATA_BOPO3A7S
#define BS_SERIALIZE_FIXDATA_BOPO3A7S

#include <boost/mpl/bool.hpp>
#include <boost/type_traits/integral_constant.hpp>

#include "bs_serialize_decl.h"

namespace blue_sky {
/*-----------------------------------------------------------------
 * special proxy object for processing invalid real numbers before saving
 *----------------------------------------------------------------*/
template< class >
struct serialize_first_fixer {
	typedef int type;
};

template< class Archive >
class BS_API_PLUGIN serialize_fix_data {

	// needed for complicated save path
	//template< class T, class fixer >
	//struct resolve_save_ret_t {

	//	template< class R, class fixer_, class is_applicable >
	//	struct peek_ret_t {
	//		typedef R type;
	//	};
	//	template< class R, class fixer_ >
	//	struct peek_ret_t< R, fixer_, boost::true_type > {
	//		typedef typename serialize_fix_applicable< R, fixer_ >::save_ret_t type;
	//	};

	//	typedef serialize_fix_applicable< T, fixer > sfa_first;
	//	typedef resolve_save_ret_t<
	//		peek_ret_t< T, fixer, typename sfa_first::on_save >,
	//		typename fixer::next
	//	> type;
	//};
	//// boundary condition for empty fixer
	//template< class T >
	//struct resolve_save_ret_t< T, int > {
	//	typedef T type;
	//};


public:
	typedef typename Archive::is_saving is_saving;
	typedef typename Archive::is_loading is_loading;
	typedef typename serialize_first_fixer< serialize_fix_data >::type first_fixer;

	serialize_fix_data(Archive& ar) : ar_(ar) {}
	serialize_fix_data(const serialize_fix_data& rhs) : ar_(const_cast< Archive& >(rhs.ar_)) {}

	// SAVE PATH -- disabled, too complex
	//// go_save() actually parse chain of fixes
	//// int is fixers chain terminator
	//template< class T >
	//static const T go_save(const T& v, const int) {
	//	return v;
	//}

	//template< class T, class fixer >
	//static const typename resolve_save_ret_t< T, fixer >::type
	//go_save(const T& v, const fixer& f) {
	//	// apply fix and select next node in chain
	//	return go_save(
	//		do_fix_save(v, f, typename serialize_fix_applicable< T, fixer >::on_save()),
	//		typename fixer::next()
	//	);
	//}

	//// do_fix_save() actually selects whether to apply fix and apply it
	//// for non-applicable types return unchanged value
	//template< class T, class fixer >
	//static const T do_fix_save(const T& v, const fixer&, const boost::false_type) {
	//	return v;
	//}

	//// specialization for applicable types
	//template< class T, class fixer >
	//static const typename serialize_fix_applicable< T, fixer >::save_ret_t
	//do_fix_save(const T& v, const fixer&, const boost::true_type) {
	//	return fixer::do_fix_save(v);
	//}

	//// overload saving operator
	//template< class T >
	//serialize_fix_data& operator <<(const T& v) {
	//	typedef typename resolve_save_ret_t< T, first_fixer >::type R;
	//	const R& fv = go_save(v, first_fixer());
	//	ar_ << fv;
	//	return *this;
	//}

	// SAVE PATH
	// go_save() actually parse chain of fixes
	// int is fixers chain terminator
	template< class Archive_, class T >
	static void go_save(Archive_& ar, const T& v, const int) {}

	template< class Archive_, class T, class fixer >
	static void go_save(Archive_& ar, const T& v, const fixer& f) {
		// apply fix
		do_fix_save(ar, v, f, typename serialize_fix_applicable< T, fixer >::on_save());
		// goto next node in chain
		go_save(ar, v, typename fixer::next());
	}

	// do_fix_save() actually selects whether to apply fix and apply it
	// for non-applicable types return unchanged value
	template< class Archive_, class T, class fixer >
	static void do_fix_save(Archive_& ar, const T& v, const fixer&, const boost::false_type) {
		// for non-applicable types just push value to archive
		ar << v;
	}

	// specialization for applicable types
	template< class Archive_, class T, class fixer >
	static void do_fix_save(Archive_& ar, const T& v, const fixer&, const boost::true_type) {
		fixer::do_fix_save(ar, v);
	}

	// overload saving operator
	template< class T >
	serialize_fix_data& operator <<(const T& v) {
		//typedef typename resolve_save_ret_t< T, first_fixer >::type R;
		//const R& fv = go_save(v, first_fixer());
		//ar_ << fv;
		go_save(ar_, v, first_fixer());
		return *this;
	}

	// LOAD PATH
	// go_load() actually parse chain of fixes
	// chain boundary operator -- does nothing
	template< class Archive_, class T >
	static void go_load(Archive_&, T&, const int) {}

	template< class Archive_, class T, class fixer >
	static void go_load(Archive_& ar, T& v, const fixer& f) {
		// apply fix
		do_fix_load(ar, v, f, typename serialize_fix_applicable< T, fixer >::on_load());
		// goto next node in chain
		go_load(ar, v, typename fixer::next());
	}

	// do_fix_load() actually selects whether to apply fix and apply it
	template< class Archive_, class T, class fixer >
	static void do_fix_load(Archive_& ar, T& v, const fixer&, const boost::false_type) {
		// for non-applicable types just read value from archive
		ar >> v;
	}

	template< class Archive_, class T, class fixer >
	static void do_fix_load(Archive_& ar, T& v, const fixer&, const boost::true_type) {
		// call fixer to load value
		fixer::do_fix_load(ar, v);
	}

	// overload loading operator
	template< class T >
	serialize_fix_data& operator >>(T& v) {
		go_load(ar_, v, first_fixer());
		return *this;
	}

	// if Archive::is_saving is false
	template< class T >
	void op_saveload(T& v, const boost::mpl::bool_< false >) {
		*this >> v;
	}

	// if Archive::is_saving is true
	template< class T >
	void op_saveload(const T& v, const boost::mpl::bool_< true >) {
		*this << v;
	}

	// overload & operator
	template< class T >
	serialize_fix_data& operator &(T& v) {
		op_saveload(v, typename Archive::is_saving());
		return *this;
	}

	// conversions
	operator Archive&() {
		return ar_;
	}

	operator const Archive&() const {
		return ar_;
	}

private:
	// store reference to original archive
	Archive& ar_;
};

} /* blue_sky */


#endif /* end of include guard: BS_SERIALIZE_FIXDATA_BOPO3A7S */

