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

#include <algorithm>
#include <iostream>

#ifdef UNIX
	#include <pthread.h>
#elif defined(_WIN32) && defined(_MSC_VER)
	#include "windows.h"
#endif

#include "bs_misc.h"
#include "bs_report.h"
#include "bs_exception.h"
#include "bs_link.h"

using namespace std;
using namespace Loki;

namespace blue_sky {
namespace bs_private {
	unsigned long int get_thread_id() {
#ifdef UNIX
		return (unsigned long int)pthread_self();
#elif defined(_WIN32) && defined(_MSC_VER)
		return (unsigned long int)GetCurrentThreadId();
#endif
	}
} // namespace bs_private

	int bs_channel::counter = 0;

	bool bs_stream::subscribe(bs_channel &src) {
		return src.attach(sp_stream(this));
	}

	bool bs_stream::unsubscribe(bs_channel &src) {
		return src.detach(sp_stream(this));
	}

	bs_channel::bs_channel()
		: output_time(false)
		, wait_end(true)
		, can_output(true)
		, buf_(new ostringstream)
	{
		char str[8];
		sprintf(str,"%d",counter);
		counter++;
		name = str;
	}

	bs_channel::bs_channel(const string &src)
		: output_time(false)
		, wait_end(true)
		, can_output(true)
		, name(src)
		, buf_(new ostringstream)
	{}

	bs_channel::bs_channel(const bs_channel &src)
		: bs_refcounter(src)
		, buf_(new ostringstream)
	{
		*this = src;
	}

	bs_channel::bs_channel(const bs_channel &src, const std::string &tname)
		: bs_refcounter(src)
		, buf_(new ostringstream)
	{
		*this = src;
		name = tname;
	}

	const bs_channel &bs_channel::operator=(const bs_channel &src) {
		msects = src.msects;
		output_time = src.output_time;
		wait_end = src.wait_end;
		can_output = src.can_output;
		name = src.name;
		buf_.lock ()->str(src.buf_.lock ()->str().c_str());
		scribes = src.scribes;
		prefix = src.prefix;
		return *this;
	}


	bool bs_channel::attach(const sp_stream &src) {
		sp_scr_list::iterator s = find(scribes.begin(),scribes.end(),src);
		if(s == scribes.end()) {
			scribes.push_back(src);
			return true;
		}
		return false;
	}

	bool bs_channel::detach(const sp_stream &src) {
		sp_scr_list::iterator s = find(scribes.begin(),scribes.end(),src);
		if (s != scribes.end()) {
			scribes.remove(src);
			return true;
		}
		return false;
	}

	void bs_channel::send_to_subscribers() {
		if (output_time && buf_.lock ()->str().length()) {
			ostringstream tmp;
			tmp << "[" << gettime() << "]: " << prefix;
			buf_.lock ()->str(tmp.str() + buf_.lock ()->str());
			//for (sp_scr_list::iterator i = scribes.begin(); i != scribes.end(); ++i)
				//(*i).lock()->write(tmp.str());
		}

		for (sp_scr_list::iterator i = scribes.begin(); i != scribes.end(); ++i)
			i->lock()->write(buf_.lock ()->str());

		buf_.lock ()->str("");
	}

	void bs_channel::set_can_output(bool t) {
		can_output = t;
	}

	void bs_channel::set_output_time() {
		if(output_time)
			output_time = false;
		else
			output_time = true;
	}

	void bs_channel::set_wait_end() {
		if(wait_end) {
			wait_end = false;
			output_time = false;
		}
		else
			wait_end = true;
	}

	bs_channel &bs_channel::add_section(int sect) {
		msects[sect];
		return *this;
	}

	bs_channel &bs_channel::rem_section(int sect) {
		msect_iterator iter = msects.find(sect);
		if (iter != msects.end())
			msects.erase(iter);
		return *this;
	}

	bs_channel &bs_channel::set_priority(priority tp) {
		msects[tp.sect] = tp.prior;
		return *this;
	}
	bool bs_channel::outputs_time() const { return output_time; }
	bool bs_channel::waits_end() const { return wait_end; }

	sp_channel bs_channel::operator<<(priority op) {
		msect_const_iterator iter = msects.find(op.sect);
		if (iter != msects.end()) {
			if (iter->second < op.prior || iter->second == -1)
				//this->send_to_subscribers();
				this->can_output = false;
			else
				this->can_output = true;
		}
		else
			this->can_output = false;
		return sp_channel(this);
	}

	void bs_channel::set_prefix(const std::string &pref) {
		prefix = pref;
	}

	sp_channel bs_channel::operator<<(sp_channel(*foo)(const sp_channel&)) {
		return foo(sp_channel(this));
	}

	void cout_scriber::write(const std::string &str) const {
//#ifdef _DEBUG
    // TODO: miryanov
    static bool is_buffer_installed = false;
    if (!is_buffer_installed)
      {
        static char cout_buffer [2*4096] = {0};
        cout.rdbuf ()->pubsetbuf (cout_buffer, sizeof (cout_buffer));
        is_buffer_installed = true;
      }

    cout << str.c_str ();
//#endif
	}

	file_scriber::file_scriber(const std::string &filename, ios_base::openmode mode)
		: file(new fstream(filename.c_str(),mode))
	{}

	//file_scriber::~file_scriber() {
	//	file.lock()->close();
	//}

	void file_scriber::write(const std::string &str) const {
#ifdef _DEBUG
    // TODO: miryanov
		*(file.lock()) << str;
#endif
	}

	namespace bs_private {

		struct BS_API log_wrapper {
			log_wrapper()	: ref_fun_(&log_wrapper::initial_log_getter) {
				//explicitly increment reference counter
				j.add_ref();
			}

			~log_wrapper() {
			}

			bs_log &usual_log_getter() {
				return j;
			}

			bs_log &initial_log_getter() {
				ref_fun_ = &log_wrapper::usual_log_getter;

				init_loging();
				return j;
			}

			void init_loging();

			bs_log& j_ref() {
				return (this->*ref_fun_)();
			}

			bs_log j;
			bs_log& (log_wrapper::*ref_fun_)();
		};

		struct BS_API thread_log_wrapper {
			thread_log_wrapper() : ref_fun_(&thread_log_wrapper::initial_log_getter) {
			}

			thread_log &usual_log_getter() {
				return j;
			}

			thread_log &initial_log_getter() {
				ref_fun_ = &thread_log_wrapper::usual_log_getter;

				init_loging();
				return j;
			}

			void init_loging();

			thread_log& j_ref() {
				return (this->*ref_fun_)();
			}

			thread_log j;
			thread_log& (thread_log_wrapper::*ref_fun_)();
		};

	}

	sp_channel bs_log::add_channel(const sp_channel &dest) {
		schan_iter_const itr = schan.find(dest);
		if (itr == schan.end()) {
			sp_channel ch_tmp = *schan.insert(dest).first;
			this->fire_signal(bs_log::log_channel_added);
			return ch_tmp;
		}
		return *itr;
	}

	bool bs_log::rem_channel(const std::string &ch_name) {
		sp_channel s(new bs_channel(ch_name));
		schan_iter itr = schan.find(s);
		if (itr != schan.end()) {
			schan.erase(itr);
			this->fire_signal(bs_log::log_channel_removed);
			return true;
		}
		return false;
	}

	std::list< std::string > bs_log::channel_list() const {
		std::list< std::string > l;
		for (schan_iter_const i = schan.begin(); i != schan.end(); ++i)
			l.push_back((*i)->get_name());
		return l;
	}

	const sp_channel& bs_log::operator[](const std::string &name_) const {
		sp_channel s(new bs_channel(name_));
		schan_iter_const itr = schan.find(s);
		if (itr != schan.end()) {
			//(*itr)->wait();
			return (*itr);
		}
		else {
			//char tmp[32];
			//     sprintf(tmp,"%d",name_);
			throw bs_exception("log",blue_sky::out_of_range,"Unknown log name",false,name_.c_str());
		}
	}

  locked_channel bs_log::get_locked (const std::string &name_)
  {
    schan_iter_const it = schan.begin (), e = schan.end ();
    for (; it != e; ++it)
      {
        if ((*it)->get_name () == name_)
          return locked_channel (*it);
      }

    throw bs_exception("log",blue_sky::out_of_range,"Unknown log name",false,name_.c_str());
  }

	typedef SingletonHolder< bs_private::log_wrapper, CreateUsingNew, PhoenixSingleton > log_holder;
	typedef SingletonHolder< bs_private::thread_log_wrapper, CreateUsingNew, PhoenixSingleton > thread_log_holder;

	template< > BS_API bs_log& singleton< bs_log >::Instance() {
		return log_holder::Instance().j_ref();
	}

	template< > BS_API thread_log& singleton< thread_log >::Instance() {
		return thread_log_holder::Instance().j_ref();
	}

	void bs_channel::dispose() const {
		delete this;
	}

	void bs_private::log_wrapper::init_loging() {
		bs_log &l = give_log::Instance();
		l.add_channel(sp_channel(new bs_channel(OUT_LOG)));
		l.add_channel(sp_channel(new bs_channel(ERR_LOG)));

		char *c_dir = NULL;
		if (!(c_dir = getenv("BS_KERNEL_DIR")))
			c_dir = (char *)".";

		string log_file = string(c_dir) + string("/blue_sky.log");

		l[OUT_LOG].lock()->attach(sp_stream(new cout_scriber));
		l[OUT_LOG].lock()->attach(sp_stream(new file_scriber(log_file,ios::out|ios::app)));
		l[OUT_LOG] << output_time;
		l[ERR_LOG].lock()->attach(sp_stream(new cout_scriber));
		log_file = string(c_dir) + string("/errors.log");
		l[ERR_LOG].lock()->attach(sp_stream(new file_scriber(log_file,ios::out|ios::app)));
		l[ERR_LOG] << output_time;

		//l[OUT_LOG] << "Output log init" << bs_end;
	}

	void bs_private::thread_log_wrapper::init_loging() {
		BSOUT << "Thread log init with address " << &j << bs_end;
	}

	sp_channel thread_log::add_log_channel(const std::string &name) {
		unsigned long int thread = bs_private::get_thread_id();
		sp_log tlog = (*logs.lock())[thread];
		if (!tlog)
			tlog = new bs_log();
		return tlog.lock()->add_channel(sp_channel(new bs_channel(name)));
	}

	sp_channel thread_log::add_log_channel(const sp_channel &ch) {
		unsigned long int thread = bs_private::get_thread_id();
		sp_log tlog = (*logs.lock())[thread];
		if (!tlog)
			tlog = new bs_log();
		return tlog.lock()->add_channel(ch);
	}

	bool thread_log::add_log_stream(const std::string &name, const sp_stream &strm) {
		unsigned long int thread = bs_private::get_thread_id();
		sp_log tlog = (*logs.lock())[thread];
		if (!tlog)
			tlog = new bs_log();
		return (*tlog.lock())[name].lock()->attach(strm);
	}

	bool thread_log::rem_log_channel(const std::string &name) {
		unsigned long int thread = bs_private::get_thread_id();
		sp_log tlog = (*logs.lock())[thread];
		if (!tlog)
			tlog = new bs_log();
		return tlog.lock()->rem_channel(name);
	}

	bool thread_log::rem_log_stream(const std::string &name, const sp_stream &strm) {
		unsigned long int thread = bs_private::get_thread_id();
		sp_log tlog = (*logs.lock())[thread];
		if (!tlog)
			tlog = new bs_log();
		return (*tlog.lock())[name].lock()->detach(strm);
	}

	void thread_log::kill() {
		unsigned long int thread = bs_private::get_thread_id();
		logs.lock()->erase(thread);
	}

	const sp_channel &thread_log::operator[](const std::string &name) {
		unsigned long int thread = bs_private::get_thread_id();
		sp_log tlog = (*logs.lock())[thread];
		if (!tlog)
			tlog = new bs_log();
		return (*tlog.lock())[name];
	}

	thread_log::thread_log()
		: logs(new mlog)
	{}

	/*thread_log::thread_log(const thread_log &tl) {
		for (const_iterator i = tl.logs.begin(); i != tl.logs.end(); ++i)
			logs.push_back(*i);
	}*/

	BS_API sp_channel bs_end(const sp_channel &r) {
		r << "\n";
		r.lock()->set_can_output(true);
		r.lock()->send_to_subscribers();
		return r;
	}
  BS_API locked_channel &bs_end (locked_channel &ch)
  {
    ch.bs_end ();
    return ch;
  }

	BS_API sp_channel output_time(const sp_channel &r) {
		r.lock()->set_output_time();
		return r;
	}

	BS_API sp_channel wait_end(const sp_channel &r) {
		r.lock()->set_wait_end();
		return r;
	}

	BS_API sp_channel bs_lock(const sp_channel &r) {
		return sp_channel(new bs_channel(*r));
	}

	BS_API sp_channel operator<<(const sp_channel &ch, sp_channel(*what)(const sp_channel&)) {
		return what(ch);
	}

    BS_API locked_channel &locked_channel::operator<< (locked_channel &(*fun) (locked_channel &))
    {
        return fun (*this);
    }

  void
  locked_channel::bs_end ()
  {
    (*locked_buf_) << "\n";

    if (ch_->output_time && (*locked_buf_).str ().length())
      {
        static string open_ = "[";
        static string close_ = "]: ";

        (*locked_buf_).str (open_ + gettime () + close_ + ch_->prefix + (*locked_buf_).str ());
      }

    std::string str = (*locked_buf_).str ();
    for (bs_channel::sp_scr_list::iterator i = ch_->scribes.begin(), e = ch_->scribes.end(); i != e; ++i)
      (*i).lock()->write (str);

    (*locked_buf_).str("");
  }

	//ctors
	/*bs_log::bs_log(const bs_log& src)
		: bs_refcounter(src), objbase(src)
	{}*/

	bs_log::bs_log()
		: bs_refcounter(), bs_messaging(BS_SIGNAL_RANGE(bs_log))
	{}

	bs_log::bs_log(const bs_log &src)
		: bs_refcounter(),bs_messaging(src)
  { schan = src.schan; }

//	bs_log::bs_log(const bs_messaging::sig_range_t &sr)
//		: bs_refcounter(), bs_messaging(sr)
//	{
//		add_signal(BS_SIGNAL_RANGE(bs_log));
//	}

	//smart pointers shouldn't delete object, because it is created as a singletone
	void bs_log::dispose() const
	{}

	// create object
  //BLUE_SKY_TYPE_STD_CREATE (bs_log);
  //BLUE_SKY_TYPE_STD_COPY (bs_log);
  //BLUE_SKY_TYPE_IMPL (bs_log, objbase, "bs_log", "Blue-Sky logger", "Class for logging blue-sky system");
}
