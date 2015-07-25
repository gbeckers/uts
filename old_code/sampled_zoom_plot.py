"""
The main executable file for the zoom_plot demo.

Right-click and drag on the upper plot to select a region to view in detail
in the lower plot.  The selected region can be moved around by dragging,
or resized by clicking on one of its edges and dragging.
"""

import wx

# Major library imports

# Traits
# Enthought imports
from enthought.enable.api import Window

# Chaco imports
from enthought.chaco.api import VPlotContainer
from enthought.chaco.tools.api import RangeSelection

# Relative imports
from old_code.grid_plot_factory import create_gridded_line_plot
from old_code.zoom_overlay import ZoomOverlay



class SampledZoomPlot(wx.Frame):

    def __init__ (self, x, y, labels={},ftitle='SignalZoomPlot',
                  *args, **kw):


        wx.Frame.__init__(self, parent=None, id=-1, size=(1200,800),
                          title=ftitle)
        self.__plots = None

        #self.SetTitle("Enable Demo")
        self.SetAutoLayout( True )
        # Create the subclass's window
        self.enable_win = self._create_window(x, y, labels=labels)
        # Listen for the Activate event so we can restore keyboard focus.
        wx.EVT_ACTIVATE( self, self._on_activate )
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.enable_win.control, 1, wx.EXPAND)
        self.SetSizer(sizer)
        self.Show( True )

    def _on_activate(self, event):
        if self.enable_win is not None and self.enable_win.control is not None:
            self.enable_win.control.SetFocus()


    def _create_window(self, x, y, labels={}):

        main_plot = create_gridded_line_plot(x,y, labels)
        zoom_plot = create_gridded_line_plot(x,y, labels)
        #zoom_plot.tools.append(PanTool(zoom_plot, constrain=True,
        #                           constrain_direction="x"))

        outer_container = VPlotContainer(padding=30,
                                         fill_padding=True,
                                         spacing=50,
                                         stack_order='top_to_bottom',
                                         bgcolor='lightgray',
                                         use_backbuffer=True)

        outer_container.add(main_plot)
        outer_container.add(zoom_plot)

        main_plot.controller = RangeSelection(main_plot)

        zoom_overlay = ZoomOverlay(source=main_plot, destination=zoom_plot)
        outer_container.overlays.append(zoom_overlay)

        return Window(self, -1, component=outer_container)

