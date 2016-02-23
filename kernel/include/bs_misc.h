/// @file
/// @author uentity, NikonovMA a ka no_NaMe <__no_name__@rambler.ru>
/// @date 12.01.2016
/// @brief Misc helper functions
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

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

// functions to convert string <-> wstring using given locale name in POSIX format
// if passed enc_name is empty, then native system locale is auto-deduced
// if enc_name = "UTF-8" or "utf-8", then country-based UTF-8 locale is used
BS_API std::string wstr2str(const std::wstring& text, const std::string& loc_name = "");
BS_API std::wstring str2wstr(const std::string& text, const std::string& loc_name = "");

// convert UTF-8 encoded string <-> string
// if passed enc_name is empty, then native system locale is auto-deduced for string
BS_API std::string ustr2str(const std::string& text, const std::string& loc_name = "");
BS_API std::string str2ustr(const std::string& text, const std::string& loc_name = "");

// generic string -> string conversion from one locale to another
// if in_enc_name is empty, assume that text given in native system locale
BS_API std::string str2str(
	const std::string& text, const std::string& out_loc_name,
	const std::string& in_loc_name = ""
);

}	//end of blue_sky namespace

#endif // __MISC_FUNCTIONS_H

