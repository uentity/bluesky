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

#include "bs_arrbase.h"
#include "bs_vecbase.h"

namespace blue_sky {

template class bs_arrbase< int >;
template class bs_arrbase< unsigned int >;
template class bs_arrbase< long >;
template class bs_arrbase< unsigned long >;
template class bs_arrbase< long long >;
template class bs_arrbase< unsigned long long >;
template class bs_arrbase< float >;
template class bs_arrbase< double >;
template class bs_arrbase< std::string >;
template class bs_arrbase< std::wstring >;

template class bs_vecbase< int >;
template class bs_vecbase< unsigned int >;
template class bs_vecbase< long >;
template class bs_vecbase< unsigned long >;
template class bs_vecbase< long long >;
template class bs_vecbase< unsigned long long >;
template class bs_vecbase< float >;
template class bs_vecbase< double >;
template class bs_vecbase< std::string >;
template class bs_vecbase< std::wstring >;

} /* namespace blue_sky */
