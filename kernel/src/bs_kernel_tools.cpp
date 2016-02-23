/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "bs_kernel_tools.h"
#include "bs_shell.h"

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

#if defined(BS_BOS_CORE_COLLECT_BACKTRACE) || defined(BS_EXCEPTION_COLLECT_BACKTRACE)
#ifdef _WIN32
#include "backtrace_tools_win.h"
#else
#include "backtrace_tools_unix.h"
#endif

std::string
kernel_tools::get_backtrace (int backtrace_depth)
{
  static const size_t max_backtrace_len = 1024;
  void *backtrace[max_backtrace_len];

  int len = tools::get_backtrace (backtrace, backtrace_depth);
  if (len)
    {
      std::string callstack = "\nCall stack: ";
      char **backtrace_names = tools::get_backtrace_names (backtrace, len);
      for (int i = 0; i < len; ++i)
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

      free (backtrace_names);
      return callstack;
    }

  return "No call stack";
}
#else
std::string
kernel_tools::get_backtrace (int backtrace_depth) {
	return "Backtrace collection disabled in this build";
}
#endif

