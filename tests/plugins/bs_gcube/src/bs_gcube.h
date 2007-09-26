#ifndef BS_GCUBE_H
#define BS_GCUBE_H

#include "base_gui.h"

namespace blue_sky {
	class BS_API_PLUGIN bs_gcube : public base_gui {
	public:
		~bs_gcube ();

		std::string gfile(const type_descriptor &td) const;

		BLUE_SKY_TYPE_DECL(bs_gcube);
	};

	namespace python {
		class BS_API_PLUGIN py_bs_gcube : public py_base_gui	{
		public:
			py_bs_gcube();
			py_bs_gcube(const py_objbase&);
			~py_bs_gcube();

			void test ();
		};
	}
}

#endif // BS_GCUBE_H
