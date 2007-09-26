import bs
import pygtk
import gtk

class cube_gui(base_gui):
    def __init__(self,cub):
        base_gui.__init__(self)
        self.baseobj = bs.bs_cube.cube(cub)
        self.gui_refresher = obj_gui_slot(self)
        self.baseobj.subscribe(bs.objbase_signal_codes.on_unlock,self.gui_refresher)
        bs.log().get("out").write("cube_gui_created")
        self.pack_start(gtk.TextView())
        self.but = gtk.Button("Cube")
        self.pack_end(self.but)

        self.but.connect('clicked',self.on_but_click)

        self.show_all()

    def refresh(self):
        bs.log().get("out").write("cube refreshing!")
        #self.baseobj.unsubscribe(2)

    def on_but_click(self,*args):
        self.baseobj.test()

#class cube_slot(bs.slot_wrap):
#    def __init__(self,obj):
#        bs.slot_wrap.__init__(self)
#        self.obj = obj

#    def execute(self,sig):
#        bs.log().get("out").write("from cube execute")
#        self.obj.feedback(sig)

def create_panel(obj):
    return cube_gui(obj)
