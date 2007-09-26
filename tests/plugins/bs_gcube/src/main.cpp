#include "stdafx.h"

#include "bs_kernel.h"
#include "bs_gcube.h"

#include "bs_plugin_common.h"

#include <boost/python/class.hpp>

namespace blue_sky {
	BLUE_SKY_PLUGIN_DESCRIPTOR_EXT("bs_gcube", "0.0.1", "BS cube GUI plugin", "GUI of blue-sky cube plugin", "bs_gcube")

	BLUE_SKY_REGISTER_PLUGIN_FUN
	{
		bool res = BLUE_SKY_REGISTER_TYPE(*bs_init.pd_, bs_gcube);
		return res;
	}

#ifdef BSPY_EXPORTING_PLUGIN
	using namespace python;
	using namespace boost::python;

	BLUE_SKY_INIT_PY_FUN {
		//BS_EXPORT_OBJBASE_CLASS_SHORT(bs_gcube,"gcube")
		class_<py_bs_gcube, bases <py_base_gui>, boost::noncopyable>("gcube")
			.def(init<const py_objbase&>())
			BS_DEF_EXPORT_SHORT2(bs_gcube,test)
		;
	}
#endif // BSPY_EXPORTING_PLUGIN
}
