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

#ifndef BS_CUBE_T_H_
#define BS_CUBE_T_H_

#include "bs_common.h"
#include "bs_cube.h"

namespace blue_sky {

	template< typename T >
	class BS_API_PLUGIN bs_cube_t : public bs_cube
	{
		template< class, template< class > class U > friend class bs_cube_tt;
	public:
		~bs_cube_t();
		void test();

	private:
		T var_;

		BLUE_SKY_TYPE_DECL_T(bs_cube_t)
	};

	//example of complicated template class
	template< class T, template< class > class U = bs_cube_t >
	class BS_API_PLUGIN bs_cube_tt : public objbase {
		typedef U< T > u_type;

	public:
		typedef smart_ptr< bs_cube_tt, true > sp_cube_tt;
		void test();

		static sp_cube_tt create(const u_type&);

	private:
		u_type u_;

		BLUE_SKY_TYPE_DECL_T(bs_cube_tt)
	};
}

// test line for test commit
#endif
