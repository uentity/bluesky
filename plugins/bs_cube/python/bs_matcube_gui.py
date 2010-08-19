import bs
import pygtk
import gtk

class matcube_gui(base_gui):
    def __init__(self,cub):
        base_gui.__init__(self)
        self.baseobj = bs.bs_matcube.cube(cub)
        self.gui_refresher = obj_gui_slot(self)
        self.baseobj.subscribe(bs.objbase_signal_codes.on_unlock,self.gui_refresher)
        bs.log().get("out").write("matcube_gui_created")
        self.pack_start(gtk.TextView())
        self.but = gtk.Button("MatCube")
        self.pack_end(self.but)

        self.but.connect('clicked',self.on_but_click)

        self.show_all()

    def refresh(self):
        bs.log().get("out").write("matcube refreshing!")
        #self.baseobj.unsubscribe(2)

    def on_but_click(self,*args):
        self.baseobj.test()

def create_panel(obj):
    return matcube_gui(obj)
