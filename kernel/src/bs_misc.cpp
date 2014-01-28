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

/*!
	\file bs_misc.cpp
  \brief Contains implimentations of some operations with loading graph and other
  \author NikonovMA a ka no_NaMe <__no_name__@rambler.ru>
 */

//this comment line explicitly added to force svn commit

#include "bs_misc.h"
#include "bs_report.h"
#include "bs_exception.h"
#include "bs_link.h"
#include "bs_config_parser.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdarg.h>
#include <map>

#ifdef UNIX
#include <errno.h>
#include <dlfcn.h>
#endif

#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/exception.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/regex.hpp>
#include <boost/locale.hpp>

#ifdef _WIN32
#include <windows.h>
#include <time.h>
#endif

#ifdef UNIX
#include <errno.h>
#endif

#include "bs_conf_reader.h"
#include "bs_kernel.h"

using namespace std;
using namespace boost;

#include <boost/xpressive/xpressive.hpp>
#include <iostream>
 
using namespace std;
using namespace boost::xpressive;
 
#define XPN_LOG_INST log::Instance()[XPN_LOG] //!< blue-sky log for error output

namespace blue_sky {

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS> load_graph;

//!	\brief compares src with string = before + doubt + after.
bool compare(const string& src , const string& dbt, const string& before, const string& after);

//const char *find_cfg_file(std::string&, std::vector<const char*>&);
//!	\brief Get leaf of src.
void get_leaf(std::string&, const std::string &);

//!	\brief get name of file in format name.expansion
void get_name(std::string&, const std::string &);

//int version_comparator(const char *, const char *);
//! \brief separator of version-strings
void version_separator(std::vector<std::string> &, const string&);

//!	\brief Makes graph with list cntr_.
blue_sky::error_code make_graph(load_graph &,v_lload &); // throw();

//!	\brief Depth search.
void graph_to_list(std::vector<int> &, const load_graph &);

//! type of graph's vertex
typedef graph_traits<load_graph>::vertex_descriptor load_vertex;
//! type of graph's index
typedef graph_traits < load_graph >::vertices_size_type size_type;
//typedef pair<lload,lload> load_arc;
//typedef pair<int,int> load_arc;
//typedef vector<load_arc> aload_arc;

/*!
	\brief verify filename by mask

	For example:\n
	mask_verify("../file.dll",".dll");
	\param filename - name of file
	\param mask - mask
	\return true if contains, false - otherwise
*/
bool mask_verify (const char * filename, const char * mask)
{
  return compare(filename,mask,"^(.*)","$");
}

/*!
	\brief Search name of file doubt in vector vec
	\param doubt - path to some dynamic library
	\param vec - vector of some files (using *.cfg files)
	\return If \a doubt \a = \a "blah/libbs_somefile.dll" and
	vec contain \a some \a string \a = \a "blah-blah/somefile.cfg"
	then will be return "blah-blah/somefile.cfg", NULL - otherwise.

 */
const char *find_cfg_file(string& doubt, std::vector<string> &vec) {
	for(unsigned int i = 0, size_ = (unsigned int)vec.size(); i < size_; ++i)
	{
		string tmp,tmp_vec,vec_name;
		get_leaf(tmp_vec,vec[i]);
		get_name(vec_name,tmp_vec);
		get_leaf(tmp,doubt);
		if(compare(tmp.c_str(),vec_name.c_str(),"^(libbs_|bs_|lib)","(.|_d.)(dll|so)$"))
			return vec[i].c_str();
	}
	return (const char*)(NULL);
}

/*!
	\param src - some string
	\param dbt - which this str may contain
	\param before - mask before dbt
	\param after - mask after dbt
	\return true - if contains, false - otherwise
 */
bool compare(const string& src , const string& dbt, const string& before, const string& after) {
	string t = string(before) + dbt + after;
	//regex expression(t);
	return regex_search(src, regex(t)) ? true : false;
}

//! \return string of chars, contains current time
std::string gettime() {
	time_t cur_time;
	char * cur_time_str = NULL;
	if(time(&cur_time))
	{
		cur_time_str = ctime(&cur_time);
		if(cur_time_str)
			cur_time_str[strlen(cur_time_str)-1] = '\0';
	}
	return cur_time_str;
}

void DumpV(const ul_vec& v, const char* pFname) {
	if(pFname) {
		ofstream fd(pFname, ios::out | ios::trunc);
		for(ul_vec::const_iterator pos(v.begin()); pos != v.end(); ++pos)
			fd << *pos << " ";
		fd << endl;
	}
	else {
		for(ul_vec::const_iterator pos(v.begin()); pos != v.end(); ++pos)
			cout << *pos << " ";
		cout << endl;
	}
}

typedef graph_traits<load_graph>::vertex_descriptor vertex_t;
typedef graph_traits<load_graph>::edge_descriptor edge_t;

void topo_sort_dfs(const load_graph& /*g*/, vertex_t u, vertex_t*& topo_order, int* mark) {
	mark[u] = 1; // 1 - посещённая вершина
	*--topo_order = u;
}

void topo_sort(const load_graph& g, vertex_t* topo_order) {
	std::vector<int> mark(num_vertices(g), 0);
	graph_traits<load_graph>::vertex_iterator vi, vi_end;

	for (boost::tie(vi, vi_end) = vertices(g); vi != vi_end; ++vi)
		if (mark[*vi] == 0)
			topo_sort_dfs(g, *vi, topo_order, &mark[0]);
}

/*!
	\brief get the ordered list of blue-sky libraries pathes

	To lib_list will be saved ordered sequence of loading libraries.\n
	If some dependencies will fail - list creation will be failed.
	\param lib_list - list of pairs - path & version
*/
void get_lib_list(list<lload> &lib_list) {
	v_lload cntr_;
	vector<int> vl;
	load_graph g(0);
	try {
		if (make_graph(g,cntr_) == blue_sky::no_library)
			BSERROR << "Libraries will not load" << bs_end;
		else
		{
			graph_to_list(vl,g);
			//v	l.resize (num_vertices(g));
			//topo	_sort(g, &vl[0] + num_vertices(g));

			lib_list.clear();
			//for(int i		= 0; i < (int)cntr_.size(); ++i)
			//	BSOUT << cnt	r_[i].first << " ___ " << cntr_[i].second << bs_end;

			for(int i = 0; i < (int)vl.size(); ++i) {
				lib_list.push_back(lload(cntr_[vl[i]]));
				BSOUT << cntr_[vl[i]].first << " ::: " << cntr_[vl[i]].second << bs_end;
			}
		}
	} catch (const bs_exception &e) {
		BSERROR << "Graph creation failed with exception! " << e.what() << bs_end;
	}
}

/*!
	This is recoursive searching of files. in the lib_dir directory.
	\param res - vector of strings for filenames
	\param what - expansion
	\param lib_dir - where to search.
	\return blue_sky::wrong_path if path lib_dir or what is wrong, or
	blue_sky::no_error - if all is ok.
 */
blue_sky::error_code search_files(vector<string> &res, const char * what, const char * lib_dir) {
	if(lib_dir == NULL) lib_dir = "./";

	try {
#ifndef UNIX
		filesystem::path plib_dir(lib_dir,filesystem::native);
#else
		filesystem::path plib_dir(lib_dir);
#endif

		// first file of directory as iterator
		for (filesystem::directory_iterator dll_dir_itr(plib_dir), end_itr; dll_dir_itr != end_itr; ++dll_dir_itr) {
			if (is_directory(*dll_dir_itr))
				continue;
			if (mask_verify(dll_dir_itr->path().string().c_str(),what)) {
				res.push_back(string(dll_dir_itr->path().string().c_str()));
			}
		}
	}
	catch(const filesystem::filesystem_error &e) {
		BSERROR << e.what() << bs_end;
		return blue_sky::wrong_path;
	}
	return blue_sky::no_error;
}

/*!
	\brief separator of version-strings
	\param res - separated by subversions version
	\param src - version string
 */
void version_separator(vector<string> &res, const string& src) {
	if (!src.size()) return;
	//if(!strcmp(src,"")) return;
	vector<string> tmp;
	string c;

	regex expression("(.*)[.](.*)");
	c = src;
	for(;;) {
		regex_split(back_inserter(tmp), c, expression);
		if(!tmp.size()) {
			regex_split(back_inserter(tmp), c, regex("(.*)"));
			res.push_back(tmp.back());
			break;
		}
		res.push_back(tmp.back());
		if(!regex_search(tmp.front(), expression)) {
			res.push_back(tmp.front());
			break;
		}
		c = tmp.front();
		tmp.clear();
	}
}

/*!
	\return -1 if left ver < right ver,\n
	0 if they are equal or \n
	1 if left ver > then right ver.
*/
int version_comparator(const string& lc, const string& rc) {
	vector<string> lres, rres;
	char *tmp;
	version_separator(lres, lc);
	version_separator(rres, rc);
	int i,j;
	long l,r;

	//unsigned int s = (unsigned int)((lres.size() < rres.size()) ? lres.size() : rres.size());

	i = (int)lres.size() - 1;
	j = (int)rres.size() - 1;
	for(; i > -1 && j > -1; --i, --j) {
		l = strtol(lres[i].c_str(), &tmp, 10);
		r = strtol(rres[j].c_str(), &tmp, 10);
		if(l == r)
			continue;
		else if(l < r)
			return (-1);
		else if(l > r)
			return 1;
	}

	if(lres.size() < rres.size())
		return (-1);
	else if(lres.size() > rres.size())
		return 1;

	return 0;
}

/*!
	\brief Search for the 'name' vertex

	Search for the full path, contains leaf == *name.* and returns index
	\param lp - vector of filenames
	\param name - name of vertex
	\return index of name or -1 - otherwise
 */
int find_vertex(vector<string> &lp, const string& name) {
	string tmp;
	for(unsigned int i = 0; i < lp.size(); ++i) {
		get_leaf(tmp, lp[i]);
		if(compare(tmp, name, "^(.*)","(_d.|.)(dll|so)$"))
			return i;
	}
	return (-1);
}

/*!
	\brief Get leaf of src.
	\param container_ - here will be leaf
	\param src - all path
 */
void get_leaf(string& container_, const string& src) {
	vector<string> res;
	string tmp = src;
	regex expression("(/\?(\?:[^\?#/]*/)*)\?([^\?#]*)");
	regex_split(back_inserter(res), tmp, expression, match_default);
	if(res.size() > 0)
		container_ = res.back();
	else
		container_.clear();
}

/*!
	\brief get name of file in format name.expansion
	\param container_ - here will be name
	\param src - all path
 */
void get_name(string &container_, const string& src) {
	vector<string> res;
	string tmp = src;
	regex expression("^([^\?#]*)[.](.*)$");
	regex_split(back_inserter(res), tmp, expression, match_default);
	if(res.size() > 0) {
		tmp = res.front();
		res.clear();
		expression = regex("^(.*)(libbs_|bs_|lib)(.*)$");
		regex_split(back_inserter(res), tmp, expression, match_default);
		if(res.size() > 0)
			container_ = res.back();
		else
			container_ = tmp;
	}
}

/*!
	\param g - here will be graph
	\param cntr_ - here will be list of pathes
	\return blue_sky::no_library - if some library in dependencies missing,
	blue_sky::no_error - otherwise.
 */
blue_sky::error_code make_graph(load_graph &g, v_lload &cntr_) {
	typedef pair<int,int> edge_t;
	typedef std::vector < edge_t > edge_array_t;

	cntr_.clear();

	int i,j;
	int k;
	sp_conf_reader cr = give_kernel::Instance().create_object(bs_conf_reader::bs_type());
	vector<string> sp, lp;
	edge_array_t edges;
	const vector<string> &c_lib_dir = bs_config ()["BLUE_SKY_PLUGINS_PATH"];

	for (size_t i = 0; i < c_lib_dir.size (); ++i) {
		search_files(sp,".cfg",c_lib_dir[i].c_str ());

#ifdef UNIX
		search_files(lp,".so",c_lib_dir[i].c_str ());
#else
		search_files(lp,".dll",c_lib_dir[i].c_str ());
		search_files(lp,".pyd",c_lib_dir[i].c_str ());
#endif
	}

	size_t lp_size = lp.size();
	for(i = 0; i < (int)lp_size; ++i) {
		cntr_.push_back(lload());
		cntr_[i].first = string(lp[i]);
	}

	for(i = 0; i < (int)lp_size; ++i) {
		char *cfg_file;
		if(!(cfg_file = (char *)find_cfg_file(lp[i],sp))) continue;

		cr.lock()->read_file(cfg_file);
		std::string msg = lp[i] + " (";
		size_t len = cr.lock()->get_length();
		for (size_t b = 0; b < len; ++b) {
			if (b != 0)
				msg += ", ";
			std::string name, ver;
			name = cr->get(b).lookup_value("name");
			ver = cr->get(b).lookup_value("version");
			msg += name;
			if(-1 != (j = find_vertex(lp,name.c_str()))) {
				edges.push_back(edge_t(i,j));
				k = version_comparator(cntr_[j].second.c_str(),ver.c_str());
				if(k == -1)
					cntr_[j].second = string(ver.c_str());
			}
			else {
			   //throw bs_exception("graph",blue_sky::no_library,name);
				if (name != "")
					BSERROR << "No library named \"" << name << "\";" << "graph creation failed!" << bs_end;
				else
					BSERROR << "You have been wrote something like \"libs: { load = ( {} ); };\"."
									<< " This is the wrong structure of config. Write \"libs: { load = ( ); };\"."
									<< "Graph creation failed!" << bs_end;
				return blue_sky::no_library;

			}
		}
#ifdef _DEBUG
		BSOUT << msg << ")" << bs_end;
#endif
	}

	try {
#if defined(BOOST_MSVC)
		g = load_graph(lp_size);
		for (std::size_t i = 0; i < edges.size (); ++i)
			add_edge(edges[i].first, edges[i].second, g);
#else // if defined(BOOST_MSVC)
		g = load_graph (edges.begin (), edges.end (), lp_size);
#endif // if defined(BOOST_MSVC)

	} catch (const std::exception &e) {
		throw bs_exception(std::string("graph builder"),std::string(e.what()));
	} catch (...) {
		throw bs_exception(std::string("graph builder"),std::string("Graph building error"));
	}

	return blue_sky::no_error;
}

/*!
	\class bs_dfs_visitor
	\ingroup loading_plugins
	\brief boost::dfs_visitor<>-legatied class for graph.
 */
class bs_dfs_visitor : public dfs_visitor< > {
public:
	/*!
		\struct bs_dfs_vs
		\ingroup loading_plugins
		\brief Enumerated vertex of graph.
	*/
	struct bs_dfs_vs {
		/*!
			\brief Constructor.
			\param nt - number in depth-search
			\param g_nt - vertex
		*/
		bs_dfs_vs(size_t nt, size_t g_nt) : n(nt), g_n(g_nt) {}
		size_t n; //!< number in depth search
		size_t g_n; //!< vertex

		bool operator<(const bs_dfs_vs &vs2) const {
			return ((n < vs2.n) ? true : false);
		}
	};

	/*!
		\brief Constructor.
		\param src - list of visitor vertexes.
	*/
	bs_dfs_visitor(vector<bs_dfs_vs> &src) : lc(src) {
		lc.clear();
	}

	/*!
		\brief depth-search one step to pushing vertex to ds-list
		\param u - vertex, which must to push
	*/
	template <typename Vertex, typename Graph>
	void discover_vertex(Vertex /*u*/, const Graph &/*g*/) const
	{}

	//! End of depth search
	template <typename Vertex, typename Graph>
	void finish_vertex(Vertex u, const Graph &/*g*/) const {
		lc.push_back(bs_dfs_vs((int)lc.size(),u));
	}

	vector<bs_dfs_vs> &lc; //!< list of visits (depth-search-list)
};

/*!
	\brief Depth search.
	\param ll - list of depth search
	\param g - graph
 */
void graph_to_list(vector<int> &ll, const load_graph &g) {
	vector<bs_dfs_visitor::bs_dfs_vs> dfs;
	bs_dfs_visitor v(dfs);

	depth_first_search(g,visitor(v));

	std::sort (dfs.begin(),dfs.end());

#ifdef _DEBUG
	for (size_t i = 0; i < dfs.size (); ++i) {
		BSOUT << dfs[i].n << "(" << dfs[i].g_n << ") " << bs_line;
    }
	BSOUT << bs_end;
#endif

	ll.clear();
	for(int i = 0; i < (int)dfs.size(); ++i)
		ll.push_back((int)dfs[i].g_n);
}

//the part from boos/filesystem/src/exception.cpp
//system error-messages
std::string system_message(int err_code) {
	string str;
#ifdef _WIN32
	LPSTR lpMsgBuf;
	::FormatMessageA(
		FORMAT_MESSAGE_ALLOCATE_BUFFER |
		FORMAT_MESSAGE_FROM_SYSTEM |
		FORMAT_MESSAGE_IGNORE_INSERTS,
		NULL,
		err_code,
		MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), // Default language
		(LPSTR)&lpMsgBuf,
		0,
		NULL
	);
	str = static_cast< const char* >(lpMsgBuf);
	::LocalFree( lpMsgBuf ); // free the buffer
	str = str.substr(0, str.find('\n'));
	str = str.substr(0, str.find('\r'));
#else
	str = ::strerror(err_code);
#endif
	return str;
}

std::string
dynamic_lib_error_message() {
#ifdef _WIN32
	return system_message (GetLastError ());
#else
	const char *msg_ = dlerror ();
	if (!msg_) {
		return std::string ("NO ERROR");
	}

	return std::string (msg_);
#endif
}


string last_system_message() {
	int err_code;
#ifdef _WIN32
	err_code = GetLastError();
#else
	err_code = errno;
#endif
	return system_message(err_code);
}


// hidden namespace
namespace {
// BS kernel locale generator
struct loc_storage {
	loc_storage()
		: native_loc(boost::locale::util::get_system_locale())
	{
		gloc.locale_cache_enabled(true);
		const std::locale& native = gloc.generate(native_loc);
		native_loc_prefix =
			std::use_facet< boost::locale::info >(native).language() + "_" +
			std::use_facet< boost::locale::info >(native).country();
		native_loc_utf8 = native_loc_prefix + ".UTF-8";
	}

	// obtain locale
	// if empty locale passed, generate native system locale
	std::locale operator()(const std::string& loc_name = "") const {
		if(loc_name.empty())
			return gloc.generate(native_loc);
		else if(loc_name == "utf-8" || loc_name == "UTF-8")
			return gloc.generate(native_loc_utf8);
		else
			return gloc.generate(loc_name);
	}

	// return UTF-8 locale with native system country settings
	std::locale native_utf8() const {
		return operator()(native_loc_utf8);
	}

	boost::locale::generator gloc;
	std::string native_loc;
	std::string native_loc_prefix;
	std::string native_loc_utf8;
};
// storage singleton
static loc_storage ls_;

} // eof hidden namespace

// functions to convert string <-> wstring
std::string wstr2str(const std::wstring& text, const std::string& loc_name) {
	return boost::locale::conv::from_utf(text, ls_(loc_name));
}

std::wstring str2wstr(const std::string& text, const std::string& loc_name) {
	return boost::locale::conv::to_utf< wchar_t >(text, ls_(loc_name));
}

std::string ustr2str(const std::string& text, const std::string& loc_name) {
	return boost::locale::conv::from_utf(text, ls_(loc_name));
}

std::string str2ustr(const std::string& text, const std::string& loc_name) {
	return boost::locale::conv::to_utf< char >(text, ls_(loc_name));
}

std::string str2str(
	const std::string& text, const std::string& out_loc_name, const std::string& in_loc_name
) {
	if(in_loc_name.size())
		return boost::locale::conv::between(text, out_loc_name, in_loc_name);
	else
		return boost::locale::conv::between(text, out_loc_name, ls_.native_loc);
}

// register misc kernel types
kernel::types_enum register_misc_types() {
	kernel::types_enum te;
	te.push_back(bs_conf_reader::bs_type());
	return te;
}

}	//end of namespace blue_sky

