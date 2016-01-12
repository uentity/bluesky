/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef PY_BS_EXPORTS_H_
#define PY_BS_EXPORTS_H_

#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/iterator.hpp>
#include <boost/python/operators.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/manage_new_object.hpp>
#include <boost/python/pure_virtual.hpp>

namespace blue_sky { namespace python {

using namespace boost;
using namespace boost::python;

}}

#endif // PY_BS_EXPORTS_H_
