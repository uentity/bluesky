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
  \file bs_common.h
  \brief Some common definitions
  \author Gagarin Alexander <gagrinav@ufanipi.ru>
*/

#ifndef _BS_COMMON_H
#define _BS_COMMON_H

#if PYTHON_VERSION > 25
#if defined(BSPY_EXPORTING) || defined(BSPY_EXPORTING_PLUGIN)
#include <boost/python/detail/wrap_python.hpp>
#endif
#endif

#include "setup_common_api.h"

//setup default plugin api
#include BS_SETUP_PLUGIN_API()

#define MSG_BUF_SIZE 4096
#define LOKI_FUNCTOR_IS_NOT_A_SMALLOBJECT

//common includes - used almost everywhere
#include <new>
#include <vector>
#include <string>
#include <iosfwd>
#include <algorithm>

#include "bs_fwd.h"
#include "smart_ptr.h"
#include "bs_type_info.h"
#include "bs_assert.h"

#include "loki/TypeManip.h"
//#include "loki/LokiTypeInfo.h"

#include <boost/noncopyable.hpp>

#include "declare_error_codes.h"

#define TO_STR(s) #s //!< To string convertion macro

//error_code
namespace blue_sky
{
	//define noncopyable interface
	typedef boost::noncopyable bs_noncopyable;

  DECLARE_ERROR_CODES (
    ((no_error,                 "no_error"))
    ((user_defined,             "user_defined"))
    ((unknown_error,            "unknown_error"))
    ((system_error,             "system_error"))
    ((wrong_path,               "wrong_path"))
    ((no_plugins,               "no_plugins"))
    ((no_library,               "no_library"))
    ((no_type,                  "no_type"))
    ((out_of_range,             "out_of_range"))
    ((not_permited_operation,   "not_permited_operation"))
    ((boost_error,              "boost_error"))
  );

	//forward declaration of kernel
	//class kernel;

	//! \defgroup loading_plugins loading plugins - classes for load plugins in blue-sky

	/*!
		\class plugin_descriptor
		\ingroup loading_plugins
		\brief Just plugin descriptor.
	*/
	struct BS_API plugin_descriptor {
	public:
		std::string name_;
		std::string version_;
		std::string short_descr_;
		std::string long_descr_;
		std::string py_namespace_;

		//! nil plugin constructor
		plugin_descriptor();

		//! ctor for searching in containers
		plugin_descriptor(const char* name);

#ifdef _DEBUG
    ~plugin_descriptor ();
#endif

		//! standard ctor for using in plugins
		plugin_descriptor(const BS_TYPE_INFO& plugin_tag, const char* name, const char* version, const char* short_descr,
			const char* long_descr = "", const char* py_namespace = "");

		//! test if this is a nil plugin
		bool is_nil() const;

		//comparision
		bool operator <(const plugin_descriptor& pd) const;
		bool operator ==(const plugin_descriptor& pd) const;

	private:
		friend class kernel;
		static unsigned int self_version();
		BS_TYPE_INFO tag_;
	};

	//additional comparison operators
	BS_API inline bool operator !=(const plugin_descriptor& lhs, const plugin_descriptor& rhs) {
		return !(lhs == rhs);
	}

	class plugin_initializer {
	private:
		friend class kernel;
		plugin_initializer();

	public:
		static unsigned int self_version();

		kernel& k_; //!< Reference to blue-sky kernel.
        plugin_descriptor const* pd_; //!< Pointer to descriptor of plugin being loaded
	};

	/*!
		\brief Blue-sky singleton template.
	 */
	template< class T >
	class singleton {
	public:
		static T& Instance();
	};

}	//end of blue_sky namespace

#define BLUE_SKY_PLUGIN_DESCRIPTOR_EXT_STATIC(name, version, short_descr, long_descr, py_namespace)   \
  namespace {                                                                                         \
    class BS_HIDDEN_API_PLUGIN _bs_this_plugin_tag_                                                   \
    {                                                                                                 \
    };                                                                                                \
  }                                                                                                   \
static const blue_sky::plugin_descriptor* bs_get_plugin_descriptor()                                  \
{                                                                                                     \
  static blue_sky::plugin_descriptor *plugin_info_ = new blue_sky::plugin_descriptor (                \
    BS_GET_TI (_bs_this_plugin_tag_),                                                                 \
    name, version, short_descr, long_descr, py_namespace);                                            \
  return plugin_info_;                                                                                \
}

/*!
	\brief Plugin descriptor macro.

	This macro should appear only once in all your plugin project, somewhere in main.cpp.
	Never put it into header file!
	Plugin descriptor generated by this macro is seeked upon plugin loading. If it isn't found,
	your library won't be recognized as BlueSky plugin, so don't forget to declare it.
	BLUE_SKY_PLUGIN_DESCRIPTOR_EXT allows you to set Python namespace (scope) name for all classes
	exported to Python.

  \param tag = tag for class
	\param name = name of the plugin
	\param version = plugin version
	\param short_descr = short description of the plugin
	\param long_descr = long description of the plugin
*/
#define BLUE_SKY_PLUGIN_DESCRIPTOR_EXT(name, version, short_descr, long_descr, py_namespace)          \
  namespace {                                                                                         \
    class BS_HIDDEN_API_PLUGIN _bs_this_plugin_tag_                                                   \
    {                                                                                                 \
    };                                                                                                \
  }                                                                                                   \
BS_C_API_PLUGIN const blue_sky::plugin_descriptor* bs_get_plugin_descriptor()                         \
{                                                                                                     \
  static blue_sky::plugin_descriptor *plugin_info_ = new blue_sky::plugin_descriptor (                \
    BS_GET_TI (_bs_this_plugin_tag_),                                                                 \
    name, version, short_descr, long_descr, py_namespace);                                            \
  return plugin_info_;                                                                                \
}

#define BLUE_SKY_PLUGIN_DESCRIPTOR(name, version, short_descr, long_descr) \
BLUE_SKY_PLUGIN_DESCRIPTOR_EXT(name, version, short_descr, long_descr, "")

//! type of get plugin descrptor pointer function
typedef blue_sky::plugin_descriptor* (*BS_GET_PLUGIN_DESCRIPTOR)();

/*!
\brief Plugin register function.

Write it into yours main cpp-file of dynamic library.
This is begining of function.\n
For example:\n
BLUE_SKY_REGISTER_PLUGIN_FUN {\n
return BLUE_SKY_REGISTER_OBJ(foo,create_foo);\n
}
*/
#define BLUE_SKY_REGISTER_PLUGIN_FUN \
BS_C_API_PLUGIN bool bs_register_plugin(const blue_sky::plugin_initializer& bs_init)

//!	type of plugin register function (in libs)
typedef bool (*BS_REGISTER_PLUGIN)(const blue_sky::plugin_initializer&);

#define BLUE_SKY_INIT_PY_FUN \
BS_C_API_PLUGIN void bs_init_py_subsystem()
typedef void (*bs_init_py_fn)();

//some common typedefs
typedef std::vector< ulong > ul_vec; //!< vector<unsigned long>
typedef blue_sky::smart_ptr< ul_vec, false > sp_ul_vec; //!< smart_ptr<vector<unsigned long> >

#endif //_BS_COMMON_H
