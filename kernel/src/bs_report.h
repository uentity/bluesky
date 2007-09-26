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

#ifndef _BS_REPORTER_H
#define _BS_REPORTER_H

#include <list>
#include <set>
#include <map>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include "bs_common.h"
#include "bs_refcounter.h"
#include "bs_object_base.h"

#include "loki/Singleton.h"

//#include <boost/thread/condition_variable.hpp>

namespace blue_sky {

	//typedef enum {
	//	OUT_LOG, //!< output to "$(BS_KERNEL_DIR)/blue_sky.log" file
	//	ERR_LOG, //!< output to "(BS_KERNEL_DIR)/errors.log" file
	//} jours;

#define OUT_LOG "out"
#define ERR_LOG "err"

	struct BS_API priority {
	  priority(int v = -1,int p = -1) : sect(v),prior(p) {}
		int sect,prior;
	};

    //there is error with smart_pointer access... I don't know why.
	class BS_API bs_stream { //: public bs_refcounter { //!!!!!!!!!!!!!!!!!!!!!!!
	public:
		virtual bool subscribe(bs_channel&);
		virtual bool unsubscribe(bs_channel&);
		virtual void write(const std::string&) const = 0; //{}//= 0;//{std::cout << "asdddd" << std::endl;}
		//virtual void dispose() const;

		virtual ~bs_stream() {}
	};

	class BS_API bs_channel : public bs_refcounter {
		friend class bs_log;
		friend struct channel_comp_names;
	public:
		typedef smart_ptr< bs_channel, true > sp_channel;
		typedef smart_ptr< bs_stream > sp_stream;
		typedef std::list< sp_stream > sp_scr_list;
		typedef std::map<int, int> msect;
		typedef msect::const_iterator msect_const_iterator;
		typedef msect::iterator msect_iterator;

		bool attach(const sp_stream&);
		bool detach(const sp_stream&);

		template <class T> sp_channel operator<<(const T &data_) {
			if (can_output) {
				//if (&data_)
				//	*(buf_.lock()) << data_;
				//else
				//	*buf_.lock() << "(nil)";
				*(buf_.lock()) << data_;
				if (!wait_end)
					send_to_subscribers();
			}
			return sp_channel(this);
		}

		bs_channel &add_section(int);
		bs_channel &rem_section(int);
		bs_channel &set_priority(priority);
		void set_can_output(bool);

		sp_channel operator<<(priority);

		sp_channel operator<<(sp_channel(*)(const sp_channel&));

		//friend struct channel_comp_names;
		friend bool operator == (const bs_channel &lc, const bs_channel &rc);
		friend bool operator == (const bs_channel &lc, const std::string &rstr);
		friend bool operator < (const bs_channel &lc, const bs_channel &rc);
		friend bool operator < (const bs_channel &lc, const std::string &rstr);

		~bs_channel() {}

		void send_to_subscribers();

    //protected:
		//! ctors
		bs_channel();
		bs_channel(const std::string&);

		//! copy ctor
		bs_channel(const bs_channel&);
		bs_channel(const bs_channel &src, const std::string &tname);

		//! Assignment operator
		const bs_channel &operator=(const bs_channel&);

		void set_output_time();
		void set_wait_end();
		bool outputs_time() const;
		bool waits_end() const;

		void set_prefix(const std::string&);

		std::string get_name() const {return name;}

		void dispose() const;

	protected:
		std::map<int,int> msects; //priority_pair rpair;
		bool output_time, wait_end, can_output;
		static int counter;
		std::string name;
		smart_ptr<std::ostringstream> buf_;
		//st_smart_ptr<std::ostringstream> buf_;
		sp_scr_list scribes;
		std::string prefix;

    friend struct locked_channel;
	};
	typedef bs_channel::sp_channel sp_channel;
	typedef bs_channel::sp_stream sp_stream;

	template<class T>
	sp_channel operator<<(const sp_channel &ch, const T &what) {
		return *(ch.lock()) << what;
	}

	BS_API sp_channel operator<<(const sp_channel &ch, sp_channel(*what)(const sp_channel&));

	struct channel_comp_names {
		bool operator()(const smart_ptr< bs_channel > &lhs, const smart_ptr< bs_channel/*, true*/ > &rhs) const {
			return (lhs->get_name() < rhs->get_name());
		}
	};

  struct BS_API locked_channel
  {
    locked_channel (const sp_channel &ch)
    : ch_ (ch)
    , locked_buf_ (ch_->buf_)
    , buf_ (*locked_buf_)
    {

    }

    void
    bs_end ();

    template <typename T>
    void
    output (const T &what)
    {
      if (ch_->can_output)
        {
          //if (&what)
            buf_ << what;
          //else
            //buf_ << "(nil)";
        }
    }


    template <typename T>
    locked_channel &operator << (const T &what)
    {
        output (what);
        return *this;
    }

    locked_channel &operator<< (locked_channel &(*fun) (locked_channel &));

    inline locked_channel &operator<< (const priority &)
    {
        return *this;
    }


    lsmart_ptr <sp_channel> ch_;
    lsmart_ptr <smart_ptr <std::ostringstream> > locked_buf_;
    std::ostringstream &buf_;
  };


	class BS_API bs_log : public bs_messaging {
		friend struct bs_private::log_wrapper;
		friend class thread_log;
		friend class std::map<int,bs_log>;
	public:
		typedef std::set< sp_channel , channel_comp_names > schannel;
		typedef schannel::iterator schan_iter;
		typedef schannel::const_iterator schan_iter_const;

		BLUE_SKY_SIGNALS_DECL_BEGIN(bs_messaging)
			log_channel_added,
			log_channel_removed,
		BLUE_SKY_SIGNALS_DECL_END

		sp_channel add_channel(const sp_channel&);
		bool rem_channel(const std::string&);

		const sp_channel& operator[](const std::string &name_) const;
    locked_channel get_locked (const std::string &channel_name);

		std::list< std::string > channel_list() const;

		void dispose() const;

		virtual ~bs_log() {}
		bs_log(const bs_log &src);
	protected:
		//bs_log(const bs_messaging::sig_range_t&);
	private:
		bs_log(); // : objbase() {}
		//BLUE_SKY_TYPE_DECL_T(bs_log)

		schannel schan;
	};

	class BS_API cout_scriber : public bs_stream {
	public:
		void write(const std::string &str) const;
	};

	class BS_API file_scriber : public bs_stream {
	public:
		typedef smart_ptr< std::fstream > sp_fstream;

		file_scriber() {}
	  file_scriber(const file_scriber &src) : bs_stream() { *this = src; }

		file_scriber(const std::string &filename, std::ios_base::openmode mode);
		//~file_scriber();
		void write(const std::string &str) const;

		file_scriber &operator=(const file_scriber &src) {
			file = src.file;
			return *this;
		}

	private:
		sp_fstream file;
	};

	class BS_API thread_log {
		friend struct bs_private::thread_log_wrapper;
	public:
		typedef smart_ptr<bs_log>               sp_log;
		typedef std::map<int, sp_log>           mlog;
		typedef smart_ptr<mlog>                 sp_mlog;
		typedef mlog::const_iterator            const_iterator;
		typedef mlog::iterator                  iterator;

		/*BLUE_SKY_SIGNALS_DECL_BEGIN(thread_log)
			log_channel_added,
			log_channel_removed,
			log_stream_added,
			log_stream_removed,
		BLUE_SKY_SIGNALS_DECL_END*/

		sp_channel add_log_channel(const std::string&);
		sp_channel add_log_channel(const sp_channel&);
		bool add_log_stream(const std::string&,const sp_stream&);
		bool rem_log_channel(const std::string&);
		bool rem_log_stream(const std::string&,const sp_stream&);

		void kill();

		const sp_channel &operator[](const std::string&);

	private:
		thread_log();
		//thread_log(const thread_log&);

		sp_mlog logs;
	};

	typedef singleton< bs_log > give_log;
	typedef singleton< thread_log > give_tlog;

	BS_API sp_channel bs_end(const sp_channel&);
  BS_API locked_channel &bs_end (locked_channel &ch);

	BS_API sp_channel output_time(const sp_channel&);

	BS_API sp_channel wait_end(const sp_channel&);

	BS_API sp_channel bs_lock(const sp_channel&);

	//BS_API const bs_channel &unlock(const bs_channel&);

	inline bool operator == (const bs_channel &lc, const bs_channel &rc) {
		return (lc.name == rc.name);
	}

	inline bool operator < (const bs_channel &lc, const bs_channel &rc) {
		return (lc.name < rc.name);
	}

	inline bool operator == (const bs_channel &lc, const std::string &rc_name) {
		return (lc.name == rc_name);
	}

	inline bool operator < (const bs_channel &lc, const std::string &rc_name) {
		return (lc.name < rc_name);
	}

}

#define BSOUT blue_sky::give_log::Instance()[OUT_LOG]
#define BSERROR blue_sky::give_log::Instance()[ERR_LOG]

#endif // _BS_REPORTER_H
