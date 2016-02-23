/// @file
/// @author Sergey Miryanov
/// @date 08.07.2009
/// @brief Tools to declare BlueSky kernel error codes
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef BS_DECLARE_ERROR_CODES_H_
#define BS_DECLARE_ERROR_CODES_H_

#ifdef BSPY_EXPORTING
#include <boost/python/enum.hpp>
#endif
#include <boost/preprocessor.hpp>

#define DECLARE_ERROR_CODE_ENUM_I(r, data, i, elem)                       \
  BOOST_PP_TUPLE_ELEM (2, 0, elem) = i,

#define DECLARE_ERROR_CODE_NAME_I(r, data, i, elem)                       \
  BOOST_PP_TUPLE_ELEM (2, 1, elem),

#define DECLARE_ERROR_CODE_PY_EXPORT_I(r, data, i, elem)                  \
  .value (BOOST_PP_STRINGIZE (BOOST_PP_TUPLE_ELEM (2, 0, elem)), BOOST_PP_TUPLE_ELEM (2, 0, elem))

#define DECLARE_ERROR_CODE_ENUM(seq)                                      \
  enum error_code                                                         \
  {                                                                       \
    BOOST_PP_SEQ_FOR_EACH_I (DECLARE_ERROR_CODE_ENUM_I, _, seq)           \
  };

#define DECLARE_ERROR_CODE_NAME(seq)                                      \
  inline std::string                                                      \
  bs_error_message (const error_code &ec)                                 \
  {                                                                       \
    static const char *error_code_name[] = {                              \
      BOOST_PP_SEQ_FOR_EACH_I (DECLARE_ERROR_CODE_NAME_I, _, seq)         \
    };                                                                    \
    return error_code_name [ec];                                          \
  }

#define DECLARE_ERROR_CODE_PY_EXPORT(seq)                                 \
  namespace python                                                        \
  {                                                                       \
    inline void                                                           \
    py_export_error_codes ()                                              \
    {                                                                     \
      using namespace boost::python;                                      \
      enum_ <error_code> ("error_code")                                   \
        BOOST_PP_SEQ_FOR_EACH_I (DECLARE_ERROR_CODE_PY_EXPORT_I, _, seq)  \
        .export_values ()                                                 \
        ;                                                                 \
    }                                                                     \
  }

#ifdef BSPY_EXPORTING
#define DECLARE_ERROR_CODES(seq)                                          \
  DECLARE_ERROR_CODE_ENUM (seq)                                           \
  DECLARE_ERROR_CODE_NAME (seq)                                           \
  DECLARE_ERROR_CODE_PY_EXPORT (seq)
#else
#define DECLARE_ERROR_CODES(seq)                                          \
  DECLARE_ERROR_CODE_ENUM (seq)                                           \
  DECLARE_ERROR_CODE_NAME (seq)
#endif

#endif // #ifndef BS_DECLARE_ERROR_CODES_H_

