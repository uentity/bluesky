from libblue_sky import *
from test_object import *
from bs_networking import *

def start_server(context):
	while (True):
		context.port = 8005
		server = HttpServer(context)
		server.start()
		return server
		

k = kernel()
k.load_plugins()

context = create_default_context()
server = start_server(context)
rm1 = TreeResourceManager(context, "/")
tree = Tree(context)

tree.mount("local:///", "/first/");
tree.mount("http://127.0.0.1:8005/", "/third/");
tree.mount("local:///", "/");

to1 = TestObject()

tree.create(to1, "/first/to1")
tree.create(to1, "/first/very/deep/folder/to1");

print to1.var
node1 = tree["/third/to1"];

for i in xrange(100):
	handle = node1.open_rw()
	to = TestObject(handle.get())
	to.var += 1
	print to.var
	handle.close()

tree.close_connections()
server.stop()
