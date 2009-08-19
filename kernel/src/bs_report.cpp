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

#include "bs_misc.h"
#include "bs_report.h"
#include "bs_exception.h"
#include "throw_exception.h"
#include "bs_link.h"
#include "bs_log_scribers.h"
#include "get_thread_id.h"

using namespace std;
using namespace Loki;

namespace blue_sky {

  namespace log {
    bool bs_stream::subscribe(bs_channel &src) {
      return src.attach(sp_stream(this));
    }

    bool bs_stream::unsubscribe(bs_channel &src) {
      return src.detach(sp_stream(this));
    }

    void
    bs_stream::add_section (int section, int level)
    {
      if (sections_.find (section) != sections_.end ())
        {
          bs_throw_exception ("Section already exists");
        }

      sections_.insert (std::make_pair (section, level));
    }
    void
    bs_stream::rem_section (int section)
    {
      section_iterator_t iter = sections_.find(section);
      if (iter != sections_.end())
        sections_.erase (iter);
    }

    void
    bs_stream::set_priority (const priority &p)
    {
      sections_[p.sect] = p.prior;
    }
  } // namespace log

	//bs_channel::bs_channel()
	//	: output_time(false)
	//	, wait_end(true)
	//	, can_output(true)
	//{
	//}

	bs_channel::bs_channel(const string &channel_name)
		: output_time(false)
		, wait_end(true)
		, name(channel_name)
	{}

	bs_channel::bs_channel(const bs_channel &src)
		: bs_refcounter(src)
	{
		*this = src;
	}

	bs_channel::bs_channel(const bs_channel &src, const std::string &tname)
		: bs_refcounter(src)
	{
		*this = src;
		name = tname;
	}

	const bs_channel &bs_channel::operator=(const bs_channel &src) {
		output_time     = src.output_time;
		wait_end        = src.wait_end;
		name            = src.name;
		output_streams_ = src.output_streams_;
		prefix          = src.prefix;

		return *this;
	}


	bool bs_channel::attach(const sp_stream &src) {
		stream_iterator_t s = find(output_streams_.begin(), output_streams_.end(), src);
		if(s == output_streams_.end()) {
			output_streams_.push_back(src);
			return true;
		}
		return false;
	}

	bool bs_channel::detach(const sp_stream &src) {
		stream_iterator_t s = find(output_streams_.begin(),output_streams_.end(),src);
		if (s != output_streams_.end()) {
			output_streams_.erase (s);
			return true;
		}
		return false;
	}

	//void bs_channel::send_to_subscribers() {
	//	if (output_time && buf_.str().length()) {
	//		ostringstream tmp;
	//		tmp << "[" << gettime() << "]: " << prefix;
	//		buf_.str(tmp.str() + buf_.str());
	//		//for (sp_scr_list::iterator i = output_streams_.begin(); i != output_streams_.end(); ++i)
	//			//(*i).lock()->write(tmp.str());
	//	}

	//	for (stream_iterator_t i = output_streams_.begin(); i != output_streams_.end(); ++i)
	//		i->lock()->write(buf_.str());

	//	buf_.str("");
	//}

	void bs_channel::set_output_time() {
		if(output_time)
			output_time = false;
		else
			output_time = true;
	}

	//void bs_channel::set_wait_end() {
	//	if(wait_end) {
	//		wait_end = false;
	//		output_time = false;
	//	}
	//	else
	//		wait_end = true;
	//}

	bs_channel &
  bs_channel::add_section(int section, int level) 
  {
    for (size_t i = 0, cnt = output_streams_.size (); i < cnt; ++i)
      {
        output_streams_[i]->add_section (section, level);
      }

		return *this;
	}

	bs_channel &bs_channel::rem_section(int section) {
    for (size_t i = 0, cnt = output_streams_.size (); i < cnt; ++i)
      {
        output_streams_[i]->rem_section (section);
      }

		return *this;
	}

	bs_channel &
  bs_channel::set_priority (const priority &tp) 
  {
    for (size_t i = 0, cnt = output_streams_.size (); i < cnt; ++i)
      {
        output_streams_[i]->set_priority (tp);
      }

		return *this;
	}

	//bool bs_channel::outputs_time() const { return output_time; }
	//bool bs_channel::waits_end() const { return wait_end; }

	//sp_channel bs_channel::operator<<(const priority &op) {
	//	section_iterator_const_t iter = sections_.find(op.sect);
	//	if (iter != sections_.end()) {
	//		if (iter->second < op.prior || iter->second == -1)
	//			//this->send_to_subscribers();
	//			this->can_output = false;
	//		else
	//			this->can_output = true;
	//	}
	//	else
	//		this->can_output = false;
	//	return sp_channel(this);
	//}

	void bs_channel::set_prefix(const std::string &pref) {
		prefix = pref;
	}

	//sp_channel bs_channel::operator<<(sp_channel(*foo)(const sp_channel&)) {
	//	return foo(sp_channel(this));
	//}

	sp_channel bs_log::add_channel(const sp_channel &dest) {
    BS_ERROR (dest, "bs_log::add_channel: dest is null");

		channel_iterator_const_t itr = channel_map_.find(dest->get_name ());
		if (itr == channel_map_.end()) {
      channel_map_.insert (std::make_pair (dest->get_name (), dest));
			this->fire_signal(bs_log::log_channel_added);
			return dest;
		}
		return itr->second;
	}

	bool bs_log::rem_channel(const std::string &ch_name) {

		channel_iterator_t itr = channel_map_.find(ch_name);
		if (itr != channel_map_.end()) {
			channel_map_.erase(itr);
			this->fire_signal(bs_log::log_channel_removed);
			return true;
		}
		return false;
	}

	void bs_channel::dispose() const {
		delete this;
	}

	sp_channel thread_log::add_log_channel(const std::string &name) {
		unsigned long int thread = detail::get_thread_id();
		sp_log tlog = (*logs.lock())[thread];
		if (!tlog)
			tlog = new bs_log();
		return tlog.lock()->add_channel(sp_channel(new bs_channel(name)));
	}

	sp_channel thread_log::add_log_channel(const sp_channel &ch) {
		unsigned long int thread = detail::get_thread_id();
		sp_log tlog = (*logs.lock())[thread];
		if (!tlog)
			tlog = new bs_log();
		return tlog.lock()->add_channel(ch);
	}

	bool thread_log::add_log_stream(const std::string &name, const sp_stream &strm) {
		unsigned long int thread = detail::get_thread_id();
		sp_log tlog = (*logs.lock())[thread];
		if (!tlog)
			tlog = new bs_log();
		return tlog.lock()->get_locked (name, __FILE__, __LINE__).get_channel ()->attach(strm);
	}

	bool thread_log::rem_log_channel(const std::string &name) {
		unsigned long int thread = detail::get_thread_id();
		sp_log tlog = (*logs.lock())[thread];
		if (!tlog)
			tlog = new bs_log();
		return tlog.lock()->rem_channel(name);
	}

	bool thread_log::rem_log_stream(const std::string &name, const sp_stream &strm) {
		unsigned long int thread = detail::get_thread_id();
		sp_log tlog = (*logs.lock())[thread];
		if (!tlog)
			tlog = new bs_log();
		return tlog.lock()->get_locked (name, __FILE__, __LINE__).get_channel ()->detach(strm);
	}

	void thread_log::kill() {
		unsigned long int thread = detail::get_thread_id();
		logs.lock()->erase(thread);
	}

	locked_channel thread_log::operator[](const std::string &name) {
		unsigned long int thread = detail::get_thread_id();
		sp_log tlog = (*logs.lock())[thread];
		if (!tlog)
			tlog = new bs_log();
		return tlog.lock()->get_locked (name, __FILE__, __LINE__);
	}

	thread_log::thread_log()
		: logs(new mlog)
	{}

	/*thread_log::thread_log(const thread_log &tl) {
		for (const_iterator i = tl.logs.begin(); i != tl.logs.end(); ++i)
			logs.push_back(*i);
	}*/

	//BS_API sp_channel wait_end(const sp_channel &r) {
	//	r.lock()->set_wait_end();
	//	return r;
	//}

	//BS_API sp_channel bs_lock(const sp_channel &r) {
	//	return sp_channel(new bs_channel(*r));
	//}

  //locked_channel &
  //locked_channel::operator () (int section, int level)
  //{
  //  bs_channel::section_iterator_const_t iter = ch_->sections_.find (section);
  //  if (iter != ch_->sections_.end ())
  //    {
  //      if (iter->second < level || iter->second == -1)
  //        {
  //          ch_->can_output = false;
  //        }
  //      else
  //        {
  //          ch_->can_output = true;
  //        }
  //    }
  //  else
  //    {
  //      ch_->can_output = false;
  //    }

  //  return *this;
  //}


	//ctors
	/*bs_log::bs_log(const bs_log& src)
		: bs_refcounter(src), objbase(src)
	{}*/

	bs_log::bs_log()
		: bs_refcounter(), bs_messaging () //(BS_SIGNAL_RANGE(bs_log))
	{}

	//bs_log::bs_log(const bs_log &src)
	//	: bs_refcounter(),bs_messaging(src)
  //{ schan = src.schan; }

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
