#include "stdafx.h"

#include "bs_kernel.h"
#include "bs_gui.h"
#include "base_gui.h"

#include "bs_plugin_common.h"

#include <boost/python/class.hpp>

//#include <QtCore/QMetaType>
//#include "py_object_base.h"

namespace blue_sky {
	BLUE_SKY_PLUGIN_DESCRIPTOR_EXT("bs_guifake", "0.0.1", "BS GUI plugin", "GUI of blue-sky plugin", "bs_guifake")

	BLUE_SKY_REGISTER_PLUGIN_FUN
	{
		bool res = BLUE_SKY_REGISTER_TYPE(*bs_init.pd_, bs_gui);
		res &= BLUE_SKY_REGISTER_TYPE(*bs_init.pd_, base_gui);
		return res;
	}

#ifdef BSPY_EXPORTING_PLUGIN
	using namespace python;
	using namespace boost::python;

	BLUE_SKY_INIT_PY_FUN {
		BS_EXPORT_OBJBASE_CLASS_SHORT(bs_gui,"gui")
			.def(init<const py_objbase&>())
			BS_DEF_EXPORT_SHORT2(bs_gui,add_pair)
		;

		BS_EXPORT_OBJBASE_CLASS_SHORT(base_gui,"basegui")
			.def(init<const py_objbase&>())
			BS_DEF_EXPORT_SHORT2(base_gui,test)
		;
	}
#endif // BSPY_EXPORTING_PLUGIN
}

//Q_DECLARE_METATYPE(blue_sky::python::py_objbase)
