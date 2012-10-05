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

#ifndef BS_SERIALIZE_FIXREAL_TYBZECMB
#define BS_SERIALIZE_FIXREAL_TYBZECMB

#include "setup_plugin_api.h"
#include <limits>
#include <boost/type_traits/is_floating_point.hpp>
#include <boost/mpl/bool.hpp>

namespace blue_sky {

/*-----------------------------------------------------------------
 * special proxy object for processing invalid real numbers before saving
 *----------------------------------------------------------------*/
template< class Archive >
class BS_API_PLUGIN fix_real {
public:
	typedef typename Archive::is_saving is_saving;
	typedef typename Archive::is_loading is_loading;

	fix_real(Archive& ar) : ar_(ar) {}
	fix_real(const fix_real& rhs) : ar_(const_cast< Archive& >(rhs.ar_)) {};

	// fix floating point types
	template< class T >
	inline T do_fix(const T& v, const boost::true_type) {
		typedef std::numeric_limits< T > nl;
#ifdef UNIX
		if(std::isnan(v))
			return 0;
		if(std::isinf(v))
			return nl::max();
#else
#include <float.h>
		if(_isnan(v))
			return 0;
		if(!_finite(v))
			return nl::max();
#endif
		if(v < nl::min())
			return nl::min();
		if(v > nl::max())
			return nl::max();
		return v;
	}

	// for non-fp types return unchanged value
	template< class T >
	inline const T& do_fix(const T& v, const boost::false_type) {
		return v;
	}

	// overload saving operator
	template< class T >
	fix_real& operator <<(const T& v) {
		const T& fv = do_fix(v, boost::is_floating_point< T >());
		ar_ << fv;
		return *this;
	}

	// if Archive::is_saving is false
	template< class T >
	void op_saveload(T& v, const boost::mpl::bool_< false >) {
		ar_ >> v;
	}

	// if Archive::is_saving is true
	template< class T >
	void op_saveload(const T& v, const boost::mpl::bool_< true >) {
		*this << v;
	}

	// overload & operator
	template< class T >
	fix_real& operator &(T& v) {
		op_saveload(v, typename Archive::is_saving());
		return *this;
	}

	// overload loading operator
	template< class T >
	fix_real& operator >>(T& v) {
		ar_ >> v;
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

#endif /* end of include guard: BS_SERIALIZE_FIXREAL_TYBZECMB */

