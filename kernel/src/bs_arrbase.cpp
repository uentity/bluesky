/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

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
