from traits.api import HasTraits, Instance
from traitsui.api import View, Item
from chaco.api import Plot, ArrayPlotData, PlotAxis, ArrayDataSource,\
     ScatterPlot, LinearMapper, DataRange1D, OverlayPlotContainer, jet
from chaco.tools.api import PanTool, ZoomTool
from enable.component_editor import ComponentEditor
from numpy import array, ones, zeros, nan, empty, max
from copy import deepcopy


def eventplot(starttimes, endtimes, indexrange=None):
    if not (starttimes < endtimes).all():
        raise ValueError("not all starttimes are smaller than endtimes")
    x = empty(3*starttimes.size,dtype=starttimes.dtype)
    x[::3] = starttimes
    x[1::3] = endtimes
    x[2::3] = nan
    y = zeros(x.size,dtype=starttimes.dtype)
    plotdata = ArrayPlotData(x=x, y=y)
    plot = Plot(plotdata)
    plot.plot(("x", "y"), type="line", color="red", line_width=10.0)
    plot.tools.append(PanTool(plot, constrain=False, constrain_direction="x"))
    plot.overlays.append(ZoomTool(plot, drag_button="right", always_on=True,
                        tool_mode="box", axis="index", enable_wheel=True))
    plot.overlays.append(PlotAxis(plot, orientation='bottom', title='Time (s)'))
    plot.y_axis.visible = False
    if not (indexrange is None):
        plot.index_range = indexrange
    return plot

def waveplot(x, y, xlabel, ylabel, title, labels={}, indexrange=None):

    labels = deepcopy(labels)

    labelnames = labels.keys()
    for key, values in labels.items():
        labels['%s_y'%key] = zeros(values.size)# * min(y)
    
    plotdata = ArrayPlotData(x=x, y=y, **labels)
    plot = Plot(plotdata)
    plot.plot(("x", "y"), type="line", color="blue")
    for i,key in enumerate(labelnames):
        color,marker= key.split('@')
        plot.plot((key,'%s_y'%key), type='scatter', color=color,
                  marker=marker, line_width=0.0,marker_size=6)
    plot.title = title
    plot.tools.append(PanTool(plot, constrain=False, constrain_direction="x"))
    plot.overlays.append(ZoomTool(plot, drag_button="right", always_on=True,
                        tool_mode="box", axis="index", enable_wheel=True))
    plot.overlays.append(PlotAxis(plot, orientation='left',title=ylabel))
    plot.overlays.append(PlotAxis(plot, orientation='bottom', title=xlabel))
    if not (indexrange is None):
        plot.index_range = indexrange
    return plot

def imageplot(img, x, y, xlabel, ylabel, title):
    plotdata = ArrayPlotData(imagedata=img)
    plot = Plot(plotdata)
    plot.img_plot("imagedata", xbounds=x, ybounds=y, colormap=jet)
    plot.overlays.append(PlotAxis(plot, orientation='left',title=ylabel))
    plot.overlays.append(PlotAxis(plot, orientation='bottom', title=xlabel))
    plot.overlays.append(ZoomTool(plot, drag_button="right",
                                      always_on=True,
                                      tool_mode="box",
                                      axis="index",
                                      enable_wheel=True))
    plot.tools.append(PanTool(plot, constrain=False,
                                  constrain_direction="x"))#, drag_button="middle"))
    return plot

class ImagePlot(HasTraits):

    plot = Instance(Plot)
    traits_view = View(Item('plot', editor=ComponentEditor(), show_label=False),
        width=500, height=500, resizable=True, buttons = ["OK"])

    def __init__(self, img, x, y, xlabel, ylabel, title):
        self.value_range = DataRange1D()
        self.value_range.set_bounds(img.min(),img.max())
        plotdata = ArrayPlotData(imagedata=img)
        self.plot = Plot(plotdata)
        self.plot.img_plot("imagedata", xbounds=x, ybounds=y, colormap=jet, 
                           value_range=self.value_range)
        self.plot.overlays.append(PlotAxis(self.plot, orientation='left',title=ylabel))
        self.plot.overlays.append(PlotAxis(self.plot, orientation='bottom', title=xlabel))
        self.plot.overlays.append(ZoomTool(self.plot, drag_button="right",
                                      always_on=True,
                                      tool_mode="box",
                                      axis="index",
                                      enable_wheel=True))
        self.plot.tools.append(PanTool(self.plot, constrain=False,
                                  constrain_direction="x"))#, drag_button="middle"))
        


class PlotView(HasTraits):

    plot = Instance(Plot)
    traits_view = View(Item('plot', editor=ComponentEditor(), show_label=False),
        width=500, height=500, resizable=True, buttons = ["OK"])

    def __init__(self, plot):
        super(PlotView, self).__init__()
        self.plot = plot

class EventPlotView(HasTraits):

    plot = Instance(Plot)
    traits_view = View(Item('plot', editor=ComponentEditor(), show_label=False),
        width=800, height=200, resizable=True, buttons = ["OK"])

    def __init__(self, waveplot):
        super(EventPlotView, self).__init__()
        self.plot = waveplot