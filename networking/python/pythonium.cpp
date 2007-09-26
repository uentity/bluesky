#include "pyheaders.h"

#include <networking/Context.h>
#include <networking/HttpServer.h>
#include <networking/Tree.h>
#include <networking/TreeResourceManager.h>

using namespace boost::python;
using namespace blue_sky;
using namespace blue_sky::networking;

class py_context
{
	ContextPtr fContext;
public:
	py_context()
		: fContext(new Context())
	{
		
	}

	py_context(ContextPtr context)
	{
		fContext = context;
	}

	ContextPtr get_context()
	{
		return fContext;
	}

	int get_port() const {return fContext->port;}
	void set_port(int port) {fContext->port = port;}
	
};

py_context create_default_context()
{
	return py_context(Context::create());
}

class py_http_server
{
	HttpServer fServer;
public:
	py_http_server(py_context context)
		: fServer(context.get_context())
	{
		
	}

	void start() {fServer.start();}
	void begin_stop(){fServer.begin_stop();}
	void end_stop() {fServer.end_stop();}
	void stop() {fServer.stop();}
};

class py_tree_resource_manager
{
	boost::shared_ptr<TreeResourceManager> fRm;
public:
	py_tree_resource_manager(py_context context, const std::string & prefix)
		: fRm(new TreeResourceManager(context.get_context(), prefix))
	{
	}
};

class py_obj_rw
{
	obj_rw fObj;
public:
	py_obj_rw(obj_rw obj)
		: fObj(obj)
	{}	

	sp_obj get()
	{
		sp_obj result = fObj.get();
		return result;
	}

	void close()
	{
		fObj.close();
	}
};

class py_obj_ro
{
	obj_ro fObj;
public:
	py_obj_ro(obj_ro obj)
		: fObj(obj)
	{}

	sp_obj get()
	{
		return fObj.get();	
	}
};

class py_tree_node
{
	TreeNode node;
public:
	py_tree_node(TreeNode node)
		: node(node)
	{}

	py_obj_rw open_rw()
	{
		return node.open_rw();
	}

	py_obj_ro open_ro()
	{
		return node.open_ro();
	}
};

class py_tree
{
	Tree fTree;
public:
	py_tree(py_context context)
		: fTree(context.get_context())
	{}

	void mount(const std::string & uri, const std::string & path)
	{
		fTree.mount(uri, path);
	}

	void create(python::py_objbase * obj, const std::string & path)
	{
		fTree.create(obj->get_sp(), path);
	}

	py_tree_node get(const std::string & path)
	{
		return fTree.get(path);
	}

	void close_connections()
	{
		fTree.close_connections();
	}
	
};



BOOST_PYTHON_MODULE(bs_networking)
{
	class_<sp_obj>("SpObj", init<sp_obj>())
	;

	class_<py_context>("Context")		
		.add_property("port", &py_context::get_port, &py_context::set_port)
	;	

	def("create_default_context", create_default_context);  

	class_<py_http_server>("HttpServer", init<py_context>())
		.def("start", &py_http_server::start)
		.def("begin_stop", &py_http_server::begin_stop)
		.def("end_stop", &py_http_server::end_stop)
		.def("stop", &py_http_server::stop)
	;

	class_<py_tree_resource_manager>("TreeResourceManager", init<py_context, std::string>())
	;

	class_<py_tree>("Tree", init<py_context>())
		.def("mount", &py_tree::mount)
		.def("create", &py_tree::create)
		.def("__getitem__", &py_tree::get)
		.def("close_connections", &py_tree::close_connections)
	;

	class_<py_tree_node>("Ref", no_init)
		.def("open_ro", &py_tree_node::open_ro)
		.def("open_rw", &py_tree_node::open_rw)
	;

	class_<py_obj_ro>("ObjRo", no_init)
		.def("get", &py_obj_ro::get)
	;

	class_<py_obj_rw>("ObjRw", no_init)
		.def("get", &py_obj_rw::get)
		.def("close", &py_obj_rw::close)
	;
}