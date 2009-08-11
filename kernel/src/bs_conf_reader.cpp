#include "bs_conf_reader.h"
#include "bs_kernel.h"
#include "bs_report.h"

#include <sstream>

namespace blue_sky {

	//	bs_conf_reader::conf_elem::value_t::value_t(const char *tname, const char *tvalue)
	//		: name (tname)
			//		, value (tvalue)
			//	{}

	void bs_conf_reader::conf_elem::add (const std::string &name, const std::string &val) {
		if (name == "" || val == "")
			BSERROR << "name or value is empty" << bs_end;
		mval[name] = val;
	}

	bs_conf_reader::bs_conf_reader(bs_type_ctor_param /*param*/)
	{}

	bs_conf_reader::bs_conf_reader(const bs_conf_reader& src)
		: bs_refcounter (src)
	{
		*this = src;
	}

	void bs_conf_reader::read_file(const char *filename) {
		//BSOUT << "opening file " << filename << bs_end;
		carray.clear();
		std::ifstream srcfile(filename);
		std::stringstream str;
		int idx = 0;

    if (srcfile.fail ())
      {
        throw bs_exception ("bs_conf_reader::read_file", std::string ("Can't open config file (") + filename + ")");
      }

		char lc = 0;
		for(; !(srcfile.fail () || (srcfile.eof()));) {
      char c = (char)srcfile.get();
			if (c == '}') {
				conf_elem ce;
				for (;;) {
					std::string name,op,value;
					str >> name;
					str >> op;
					str >> value;
					if (op == "=") {
						if (name == "" || value == "")
							BSERROR << "No name or value" << bs_end;
						ce.add(name,value);
						//BSOUT << name << '=' << value << bs_end;
					} else if (op == "") {
						str.clear();
						break;
					} else {
						BSERROR << "No " << op << " operator's handler yet." << bs_end;
					}
				}
				carray.push_back(ce);
			} else if (c == ' ' || c == '\n' || c == '\t' 
								 || c == ',' || c == ';' || c == '{'
								 || c == ',' || c == '\'' || c == '"') {
				if ((str.str().length() ? str.str()[str.str().length() - 1] : (char)0) != ' ')
					str << ' ';
			} else {
				str << c;
			}
		}

		//BSOUT << "len = " << carray.size() << bs_end;

	}

	size_t bs_conf_reader::get_length() const {
		return carray.size();
	}

	const bs_conf_reader::conf_elem &bs_conf_reader::get(int i) const {
		return carray[i];
	}

	bs_conf_reader::conf_elem &bs_conf_reader::get(int i) {
		return carray[i];
	}

	std::string bs_conf_reader::conf_elem::lookup_value(const std::string &what) const {
		val_map_t::const_iterator i = mval.find(what);
		if (mval.end() != i)
			return i->second;
		return "";
	}


	BLUE_SKY_TYPE_STD_COPY(bs_conf_reader);
	BLUE_SKY_TYPE_STD_CREATE(bs_conf_reader);
	BLUE_SKY_TYPE_IMPL_SHORT(bs_conf_reader, objbase, "The blue-sky config file reader");

}
