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

//void py_export_common();
void py_export_typed();
void py_export_combase();
void py_export_kernel();
void py_export_abstract_storage();
void py_export_link();
void py_export_log();
void py_export_messaging();
void py_export_objbase();
void py_export_shell();
void py_export_tree();

}}

#endif // PY_BS_EXPORTS_H_
