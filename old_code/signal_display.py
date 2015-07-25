import numpy
import wx

from enthought.chaco.api import ArrayDataSource, DataRange1D, LinePlot, \
     FilledLinePlot, BarPlot, ScatterPlot, LinearMapper, add_default_grids, PlotAxis, \
     GridPlotContainer, ArrayPlotData, create_scatter_plot, ArrayPlotData, jet, Plot, \
     GridDataSource, DataRange2D, GridMapper, ImageData, ImagePlot
from enthought.chaco.tools.api import PanTool, ZoomTool, BroadcasterTool
from enthought.enable.api import Window
from enthought.kiva import CompiledPath


from enthought.traits.api import Trait, Callable


class SignalPlot(wx.Frame):

    def __init__ (self, data, shape, ftitle='Signal Plot',
                  *args, **kw):

        self.__data = self._check_data(data)
        self.__xmapper = self._get_xmapper()
        self.__ymapper = self._get_ymapper()

        wx.Frame.__init__(self, parent=None, id=-1, size=(1200,800),
                          title=ftitle)
        self.__plots = None

        #self.SetTitle("Enable Demo")
        self.SetAutoLayout( True )
        # Create the subclass's window
        self.enable_win = self._create_window()
        # Listen for the Activate event so we can restore keyboard focus.
        wx.EVT_ACTIVATE( self, self._on_activate )
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.enable_win.control, 1, wx.EXPAND)
        self.SetSizer(sizer)
        self.Show( True )

    data = property(
        lambda self: (self.__data), None, None,
        "signal data")

    xmapper = property(
        lambda self: (self.__xmapper), None, None,
        "")
    ymapper = property(
        lambda self: (self.__ymapper), None, None,
        "")
    
    plot = property(
        lambda self: (self.__plots), None, None,
        "plots")

    def _on_activate(self, event):
        if self.enable_win is not None and self.enable_win.control is not None:
            self.enable_win.control.SetFocus()

    def _check_data(self, data):
        pass

    def _check_nchannels(self, data):
        pass

    def _iter_datasource(self):
        pass

    def _get_xmapper(self):
        pass

    def _get_ymapper(self):
        pass

    def _get_plot(self):
        pass

    def _create_window(self):

        container = GridPlotContainer(shape=self.__shape, spacing=(1.0,1.0),
                                      padding=(40,10,10,40))

        xmapper = self._get_xmapper()
        ymapper = self._get_ymapper()

        for i, data in enumerate(self._iter_datasource()):

            p = self._get_plot(data)
            add_default_grids(p)
            p.tools.append(PanTool(p, constrain=False,
                                   constrain_direction="x"))
            p.overlays.append(ZoomTool(p, drag_button="right",
                                       always_on=True,
                                       tool_mode="box",
                                       axis="index",
                                       enable_wheel=True))
            if i == 0:
                p.overlays.append(PlotAxis(p, orientation='left',title='Potential'))
            if i == (self.__nchannels - 1):
                p.overlays.append(PlotAxis(p, orientation='bottom', title='Time'))
            container.add(p)
            self.__plots.append(p)

        return Window(self, -1, component=container)