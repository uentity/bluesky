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

#include "bs_kernel_tools.h"
#include "bs_shell.h"

#ifdef _WIN32
#include "backtrace_tools_win.h"
#else
#include "backtrace_tools_unix.h"
#endif

#include <sstream>

using namespace std;
using namespace blue_sky;

struct measure_time {
	measure_time(bool go = true) {
		if(go) start();
	}

	void start() {
		sec_ = 0;
		c_ = clock();
	}

	void stop() {
		clock_t cur = clock();
		sec_ += (double)(cur - c_)/CLOCKS_PER_SEC;
	}

	double elapsed() const {
		return sec_;
	}

	ostream& operator<<(ostream& os) const {
		os << "Time elapsed: " << sec_ << " seconds" << endl;
		return os;
	}

private:
	clock_t c_;
	double sec_;
};

std::string kernel_tools::print_loaded_types() {
	ostringstream outs;
	kernel& k = give_kernel::Instance();
	outs << "------------------------------------------------------------------------" << endl;
	outs << "List of loaded BlueSky types {" << endl;
	kernel::plugins_enum pg = k.loaded_plugins();
	kernel::types_enum tp;
	for(kernel::plugins_enum::const_iterator p = pg.begin(), p_end = pg.end(); p != p_end; ++p) {
		outs << "Plugin: " << p->name_ << ", version " << p->version_ << " {" << endl;
		tp = k.plugin_types(*p);
		for(kernel::types_enum::const_iterator t = tp.begin(), t_end = tp.end(); t != t_end; ++t) {
			outs << "	" << t->stype_ << " - " << t->short_descr_ << endl;
		}
		outs << "	}" << endl;
	}

	outs << "} end of BlueSky types list" << endl;
	outs << "------------------------------------------------------------------------" << endl;
	return outs.str();
}

std::string kernel_tools::walk_tree(bool silent) {
	ostringstream outs;
	kernel& k = give_kernel::Instance();
	outs << "------------------------------------------------------------------------" << endl;
	outs << "BlueSky tree contents {" << endl;

	sp_link leaf = k.bs_root();
	if(!silent) outs << leaf->name() << endl;
	sp_node n = leaf->node();

	deep_iterator di;
	measure_time tm;
	while(!di.is_end()) {
		if(!silent) {
	//		outs << "|" << endl;
	//		outs << "+--" << di->name();
			outs << di.full_name();
			if(di->is_node())
				outs << "(+)";
			outs << '(' << di->refs() << ')' << endl;
		}
		++di;
	}
	tm.stop();

//	for(bs_node::n_iterator ni = n->begin(), end = n->end(); ni != end; ++ni) {
//		outs << "|" << endl;
//		outs << "+--" << ni->name();
//		if(ni->is_node())
//			outs << "(+)";
//		outs << endl;
//	}

	outs << "} end of BlueSky tree contents" << endl;
	outs << "Tree walking took " << tm.elapsed() << " seconds" << endl;
	outs << "------------------------------------------------------------------------" << endl;
	return outs.str();
}

std::string kernel_tools::print_registered_instances() {
	ostringstream outs;
	outs << "------------------------------------------------------------------------" << endl;
	outs << "List of BlueSky registered instances {" << endl;

	typedef vector< type_tuple > types_enum;
	types_enum all_types = BS_KERNEL.registered_types();
	sp_obj obj;
	sp_inode i;
	for(types_enum::const_iterator p = all_types.begin(), end = all_types.end(); p != end; ++p) {
		for(bs_objinst_holder::const_iterator o = BS_KERNEL.objinst_begin(p->td_),
				end_o = BS_KERNEL.objinst_end(p->td_); o != end_o; ++o) {
			obj = *o;
			i = obj->inode();
			outs << obj->bs_resolve_type() << '(' << obj.refs() << ") at " << obj.get();
			if(i) {
				outs << " -->" << endl;
				for(bs_inode::l_list::const_iterator pl = i->links_begin(), end_l = i->links_end(); pl != end_l; ++pl)
					outs << "\t'" << (*pl)->name() << "' [ bs_link(" << pl->refs() << ") at " << pl->get() << " ]" << endl;
			}
			else outs << " (dangling)" << endl;
			//outs << endl;
		}
	}
	outs << "} end of BlueSky registered instances list" << endl;
	outs << "------------------------------------------------------------------------" << endl;
	return outs.str();
}

std::string
kernel_tools::get_backtrace (size_t backtrace_depth)
{
  static const size_t max_backtrace_len = 1024;
  void *backtrace[max_backtrace_len];

  size_t len = tools::get_backtrace (backtrace, backtrace_depth);
  if (len)
    {
      std::string callstack = "\nCall stack: ";
      char **backtrace_names = tools::get_backtrace_names (backtrace, len);
      for (size_t i = 0; i < len; ++i)
        {
          if (backtrace_names[i] && strlen (backtrace_names[i]))
            {
              callstack += (boost::format ("\n\t%d: %s") % i % backtrace_names[i]).str ();
            }
          else
            {
              callstack += (boost::format ("\n\t%d: <invalid entry>") % i).str ();
            }
        }

      return callstack;
    }

  return "No call stack";
}

