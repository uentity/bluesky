#ifndef BLUE_SKY_BASE_GUI_H
#define BLUE_SKY_BASE_GUI_H

#include "bs_object_base.h"
#include "bs_command.h"

#include "py_bs_object_base.h"

namespace blue_sky {
	class BS_API_PLUGIN base_gui : public objbase {
	public:
		virtual ~base_gui ();

		virtual std::string gfile(const type_descriptor &td) const;

		BLUE_SKY_TYPE_DECL(base_gui);
	};

	namespace python {
		class BS_API_PLUGIN py_base_gui : public py_objbase	{
		public:
			py_base_gui();
			py_base_gui(sp_obj sp_obj_);
			py_base_gui(const py_objbase&);
			~py_base_gui();

			void test ();
		};
	}
}

#endif // BLUE_SKY_BASE_GUI_H
