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

/*!
  \file bs_misc.h
  \brief Contains some operations with loading graph and other
  \author NikonovMA a ka no_NaMe <__no_name__@rambler.ru>
 */
#ifndef __MISC_FUNCTIONS_H
#define __MISC_FUNCTIONS_H

#include "bs_common.h"
#include <list>
#include <clocale>
#include <locale>
#include <string>

namespace blue_sky {

//! this manipulator skips line
template <class charT, class traits>
inline
std::basic_istream <charT, traits>&
ignoreLine(std::basic_istream <charT, traits>& strm)
{
	strm.ignore(0x7fff, strm.widen('\n'));
	return strm;
}

//! verify filename by mask
BS_C_API bool mask_verify (const char * filename, const char * mask);

//! remove excess copies of T in list
template <class T>
void remove_excess (std::list <T*> *l)
{
  typedef typename std::list<T*>::iterator iterator;
  for (iterator i = l->begin (); i != l->end (); ++i)
    for (iterator j = --(l->end ()); i != j && j != l->begin (); --j)
    {
      if ((*i)->type_id () == (*j)->type_id ())
      {
        l->erase (j);
        j = l->end ();
      }
    }
}

//misc functions
BS_C_API void DumpV(const ul_vec& v, const char* pFname = NULL);

//! Type of loading plugin pair. first: plugin's path, second: plugin's version
typedef std::pair<std::string,std::string> lload;
//! type of vector of plugin pairs
typedef std::vector<lload> v_lload;

 //! \brief get the ordered list of blue-sky libraries pathes
BS_C_API void get_lib_list(std::list<lload> & //!< list of loading
							);
//! \brief string-based versions comparator
BS_C_API int version_comparator(const std::string& //!< left version
								, const std::string& //!< right version
								);
//! \brief get time function
BS_API std::string gettime();

BS_API blue_sky::error_code search_files(std::vector<std::string> &res, const char * what, const char * lib_dir);

BS_API std::string system_message(int err_code);
BS_API std::string last_system_message();

/**
 * \brief return dynamic lib error message
 * */
BS_API std::string dynamic_lib_error_message ();

// functions to convert string <-> wstring
BS_API std::string wstr2str(const std::wstring& text);
BS_API std::wstring str2wstr(const std::string& text);

}	//end of blue_sky namespace

#endif // __MISC_FUNCTIONS_H

