import numpy
import wx

from enthought.traits.api import Range
from enthought.chaco.api import ArrayDataSource, DataRange1D, LinePlot, \
     FilledLinePlot, BarPlot, ScatterPlot, LinearMapper, add_default_grids, PlotAxis, \
     GridPlotContainer, ArrayPlotData, create_scatter_plot, ArrayPlotData, jet, Plot, \
     GridDataSource, DataRange2D, GridMapper, ImageData, ImagePlot, OverlayPlotContainer, \
     DataLabel
from enthought.chaco.tools.api import PanTool, ZoomTool, BroadcasterTool, \
     LineInspector, PointMarker
from enthought.enable.api import Window
from enthought.kiva import CompiledPath

#from uts.core.uniformtime import MultiChannelUniformTimeSeries
from enthought.traits.api import Trait, Callable


class MultiChannelPlot(wx.Frame):

    def __init__ (self, data, shape, geometry=None, labels=None,
                  ftitle='Multi-Channel View', xmapper=None, ymapper=None,
                  range2d=None, xlabel='', ylabel='', time=None, *args, **kw):

        self.__data = self._check_data(data)
        self.__nchannels = self._check_nchannels()
        if xmapper is None:
            self.__xmapper = self._get_xmapper()
        else:
            self.__xmapper = xmapper
        if ymapper is None:
            self.__ymapper = self._get_ymapper()
        else:
            self.__ymapper = ymapper
        self.__geometry = geometry
        if self.__geometry is not None:
            self.__shape = (self.__geometry.ny, self.__geometry.nx)
        else:
            self.__shape = shape
        self.__xlabel = xlabel
        self.__ylabel = ylabel
        self.__labels = self._check_labels(labels)
        self.__time = Range(low=float(self.__data.starttime), high=float(self.__data.endtime),
                            value=float(self.__data.starttime))
        wx.Frame.__init__(self, parent=None, id=-1, size=(1200,800),
                          title=ftitle)
        self.__plots = []

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
        "multi-channel data")
    nchannels = property(
        lambda self: (self.__nchannels), None, None,
        "number of channels")
    xmapper = property(
        lambda self: (self.__xmapper), None, None,
        "")
    ymapper = property(
        lambda self: (self.__ymapper), None, None,
        "")
    shape = property(
        lambda self: (self.__shape), None, None,
        "shape")
    geometry = property(
        lambda self: (self.__geometry), None, None,
        "")
    plots = property(
        lambda self: (self.__plots), None, None,
        "plots")
    labels = property(
        lambda self: (self.__labels), None, None,
        "labels")
    xlabel = property(
        lambda self: (self.__xlabel), None, None,
        "xlabel")
    ylabel = property(
        lambda self: (self.__ylabel), None, None,
        "ylabel")



    def _on_activate(self, event):
        if self.enable_win is not None and self.enable_win.control is not None:
            self.enable_win.control.SetFocus()

    def _check_data(self, data):
        pass

    def _check_nchannels(self, data):
        pass

    def _iter_datasource(self):
        pass

    def _check_labels(self, labels):
        if labels is None:
            return None
        l = {}
        for key, item in labels.items():
            l[key] = (ArrayDataSource(numpy.array(item)),
                      ArrayDataSource(numpy.ones(len(item))))
        return l

    def _get_xmapper(self):
        pass

    def _get_ymapper(self):
        pass

    def _get_plot(self):
        pass

    def _create_window(self):
        # padding (l,r,t,b)
        container = GridPlotContainer(shape=self.shape, spacing=(2.0,2.0),
                        padding=(50,10,20,40),  bgcolor = "white",#"lightgray",
                        use_backbuffer=True)


        for i, data in enumerate(self._iter_datasource()):

            ol =  OverlayPlotContainer(padding=0, fill_padding = True,
                                         use_backbuffer=True)
            p = self._get_plot(data)
            add_default_grids(p)
            p.tools.append(PanTool(p, constrain=False,
                                   constrain_direction="x"))
            p.overlays.append(ZoomTool(p, drag_button="right",
                                       always_on=True,
                                       tool_mode="box",
                                       axis="index",
                                       enable_wheel=True))
            p.overlays.append(LineInspector(p, axis="index",
                                                write_metadata=True,
                                                is_listener=True, color='magenta',
                                                interactive=False))
            #p.tools.append(PointMarker(p))
            if i == 0:
                p.overlays.append(PlotAxis(p, orientation='left',title=self.ylabel))
            if i == (self.__nchannels - 1):
                p.overlays.append(PlotAxis(p, orientation='bottom', title=self.xlabel))

            if (i % self.shape[1]) == 0:
                p.y_axis.visible = True
            else:
                p.y_axis.visible = False
            if i >= self.nchannels - self.shape[1]:
                p.x_axis.visible = True
            else:
                p.x_axis.visible = False

            # labels
            colors = ['blue','red', 'green']

            if self.labels is not None:
                signal = ArrayDataSource(numpy.linspace(0.0,1.2,10))
                ymapper = LinearMapper(range=DataRange1D(signal))
                for i, (key, (labeltimes, zeros)) in enumerate(self.labels.items()):
                    signal = ArrayDataSource(numpy.linspace(0.0,1.0,10))
                    ymapper = LinearMapper(range=DataRange1D(signal))
                    sp = ScatterPlot(index = labeltimes, value = zeros,
                           index_mapper = LinearMapper(range=self.xmapper.range),
                           value_mapper = ymapper,
                           color=colors[i], marker='inverted_triangle', line_width=0.0,
                           marker_size=8)
                    ol.add(sp)
            ol.add(p)
            container.add(ol)
            self.__plots.append(p)

        return Window(self, -1, component=container)

    def set_data(self, signalarray):
        assert signalarray.nchannels == len(self.__plots),\
                "number of signal channels does not match that of plot"
        self.__signalarray = sa = signalarray
        time = sa.samplingtimes()
        for i, p in enumerate(self.__plots):
            p.index.set_data(time)
            p.value.set_data(sa.samples[:,i])
            p.request_redraw()

    def draw_pdf(self, filename, size=[600,800],
                 dest_box=(0.1, 0.1, -0.1, -0.1), units='inch'):

        from enthought.chaco.pdf_graphics_context import PdfPlotGraphicsContext
        container = self.enable_win.component
        oldbounds = container.outer_bounds
        container.outer_bounds = size
        container.do_layout(force=True)
        gc = PdfPlotGraphicsContext(filename=filename, dest_box=dest_box,
                                    dest_box_units=units)
        gc.render_component(container)
        gc.save()
        container.outer_bounds = oldbounds

    def add_label(self, x, y, color='red'):

        for plot in enumerate(self.plots):
            dl = DataLabel(component=plot, data_point=(x, y),
                           label_position="bottom right",
                           border_visible=False,
                           bgcolor="transparent",
                           marker_size=3,
                           marker_color=color,
                           marker_line_color=color,
                           marker = "pixel",
                           arrow_visible=True,
                           arrow_color=color,
                           label_format = '')
            plot.overlays.append(dl)
            plot.request_redraw()


class MultiChannelPlotP(MultiChannelPlot):

    def __init__(self, mcppes, plotparams={}):
        '''
        We expect data to be a MultiChannelPointProcessEventSet
        '''

        self.__plotparams = plotparams # a dictionary of plot parameters

        MultiChannelPlot.__init__(self, data=mcppes, shape=mcppes.mcshape)

    def _check_data(self, data):
        #assert isinstance(data, MultiChannelPointProcessEventSet)
        return data

    def _check_nchannels(self):
        return self.data.nchannels

    def _iter_datasource(self):
        for i in range(self.nchannels):
            mcppes = self.data
            origin = mcppes.origintime
            ctimes = numpy.array([],dtype=mcppes.data[i][0].dtype)
            cseqno = numpy.array([],dtype=numpy.int)
            for j,el in enumerate(mcppes.data[i]):
                ctimes = numpy.concatenate([ctimes, el-mcppes.eventstarttimes[j]-origin])
                cseqno = numpy.concatenate([cseqno, numpy.repeat(j,el.size)])
            ctimes = ArrayDataSource(ctimes)
            cseqno = ArrayDataSource(cseqno)

            yield ctimes, cseqno

    def _get_xdatasource(self):
        mcppes = self.data
        dur = mcppes.duration
        origin = mcppes.origintime
        return ArrayDataSource(numpy.linspace(0.,dur,10)-origin)

    def _get_xmapper(self):
        time = self._get_xdatasource()
        return LinearMapper(range=DataRange1D(time))

    def _get_ymapper(self):
        mcppes = self.data
        eventnos = ArrayDataSource(numpy.arange(mcppes.nevents,0,-1))
        return LinearMapper(range=DataRange1D(eventnos), low=1, high=mcppes.nevents)

    def _get_plot(self, data):
        xdatasource, ydatasource = data
        return ScatterPlot(index = xdatasource, value = ydatasource,
                           index_mapper = LinearMapper(range=self.xmapper.range),
                           value_mapper = LinearMapper(range=self.ymapper.range),
                           **self.__plotparams)

class EpochedMultiChannelPointProcessPlot(MultiChannelPlot):

    def __init__(self, mcppes, geometry=None, plotparams={}):
        '''
        We expect data to be a EpochedMultiChannelPointProcess
        '''

        plotparams['line_width'] = 0.0
        plotparams['marker_size'] = 1.0
        plotparams['bgcolor']= 'white'
        plotparams['border_visible'] = True

        self.__plotparams = plotparams # a dictionary of plot parameters

        MultiChannelPlot.__init__(self, data=mcppes, geometry=geometry,
                                  shape=mcppes.mcshape)

    def _check_data(self, data):
        #assert isinstance(data, MultiChannelPointProcessEventSet)
        return data

    def _check_nchannels(self):
        return self.data.nchannels

    def _iter_datasource(self):

        if (self.geometry is not None ) and (
            self.data.channelnames is not None ):
            channels = self.geometry.mapfrom1Dto1D(
                array=numpy.arange(self.nchannels, dtype=numpy.int),
                pointnames = self.data.channelnames)
        else:
            channels = numpy.arange(self.nchannels)
        mcppes = self.data
        origin = mcppes.origintime
        for i in channels:
            ctimes = numpy.array([],dtype=mcppes.dtype)
            cseqno = numpy.array([],dtype=numpy.int)
            for j,el in enumerate(mcppes.timings[i]):
                ctimes = numpy.concatenate([ctimes, el.timings])
                cseqno = numpy.concatenate([cseqno, numpy.repeat(j,el.npoints)])
            ctimes = ArrayDataSource(ctimes)
            cseqno = ArrayDataSource(cseqno)

            yield ctimes, cseqno

    def _get_xdatasource(self):
        mcppes = self.data
        dur = mcppes.duration
        origin = mcppes.origintime
        return ArrayDataSource(numpy.linspace(0.,dur,10)-origin)

    def _get_xmapper(self):
        time = self._get_xdatasource()
        return LinearMapper(range=DataRange1D(time))

    def _get_ymapper(self):
        mcppes = self.data
        eventnos = ArrayDataSource(numpy.arange(mcppes.nepochs,0,-1))
        return LinearMapper(range=DataRange1D(eventnos), low=1, high=mcppes.nepochs)

    def _get_plot(self, data):
        xdatasource, ydatasource = data
        return ScatterPlot(index = xdatasource, value = ydatasource,
                           index_mapper = LinearMapper(range=self.xmapper.range),
                           value_mapper = LinearMapper(range=self.ymapper.range),
                           **self.__plotparams)


#class ErpSetPlotSpikes(MultiChannelPlotP):

    #def __init__(self, mcppes):

        ##plotparams['marker'] = 'custom'
        ##plotparams['custom_symbol'] = self._make_custom_marker()

        #ErpSetPlotP.__init__(self, mcppes, plotparams=plotparams)

    #def _make_custom_marker(self):
        #path = CompiledPath()
        #path.move_to(0,-1)
        #path.line_to(0,1)
        #return path


class MultiChannelTimeSampledSignalPlot(MultiChannelPlot):

    plottypes = {'lineplot': LinePlot, 'filledlineplot': FilledLinePlot}

    def __init__(self, data, plottype='lineplot', mcshape=None, geometry=None,
                 plotparams={}, xmapper=None, ymapper=None, labels=None):
        '''
        We expect data to be a MultiChannelTimeSampledSignal.
        '''


        self.__plottype = self.plottypes[plottype]
        plotparams['border_visible'] = True
        self.__plotparams = plotparams # a dictionary of plot parameters

        if mcshape is None:
            mcshape = data.mcshape

        MultiChannelPlot.__init__(self, data, shape=mcshape, geometry=geometry,
                    xmapper=xmapper, ymapper=ymapper, labels=labels,
                    xlabel='Time (s)', ylabel='Amplitude')


    def _check_data(self, data):

        #assert isinstance(data, tsig.MultiChannelTimeSampledSignal)
        #print type(data)
        return data

    def _check_nchannels(self):
        return self.data.nchannels

    def _iter_datasource(self):
        x =  ArrayDataSource(self.data.samplingtimes())
        if (self.geometry is not None ) and (
            self.data.channelnames is not None ):
            channels = self.geometry.mapfrom1Dto1D(
                array=numpy.arange(self.nchannels, dtype=numpy.int),
                pointnames = self.data.channelnames)
        else:
            channels = numpy.arange(self.nchannels)
        for i in channels:
            yield x, ArrayDataSource(self.data[:,i].samples.get(copy=True))

    def _get_xmapper(self):
        time = ArrayDataSource(self.data.samplingtimes())
        return LinearMapper(range=DataRange1D(time))

    def _get_ymapper(self):
        samples = self.data.samples.get()
        #chind = (samples.max() - samples.min()).argmax()
        signal = ArrayDataSource(numpy.linspace(samples.min(),samples.max(),10))
        return LinearMapper(range=DataRange1D(signal))

    def _get_plot(self, data):
        xdatasource, ydatasource = data
        p = self.__plottype(index = xdatasource, value = ydatasource,
                               index_mapper = LinearMapper(range=self.xmapper.range),
                               value_mapper = LinearMapper(range=self.ymapper.range),
                               **self.__plotparams)
        p.overlays.append(PlotAxis(p, orientation='left'))
        p.overlays.append(PlotAxis(p, orientation='bottom'))
        return p

class ErpPlotLFP(MultiChannelTimeSampledSignalPlot):

    def __init__(self, data):

        plottype = FilledLinePlot
        plotparams = {'edge_color': 'blue', 'face_color': 'paleturquoise',
                      'bgcolor': 'white', 'border_visible': True}
        MultiChannelTimeSampledSignalPlot.__init__(self, data=data, plottype=plottype,
                          plotparams=plotparams)

class ErpPlotRMS(MultiChannelTimeSampledSignalPlot):

    def __init__(self, data):

        plottype = LinePlot
        plotparams = {'bgcolor': 'white', 'border_visible': True}
        MultiChannelTimeSampledSignalPlot.__init__(self, data=data, plottype=plottype,
                          plotparams=plotparams)

class ErpPlotPSTH(MultiChannelTimeSampledSignalPlot):

    def __init__(self, data):

        plottype = LinePlot
        plotparams = {'bgcolor': 'white', 'border_visible': True}
        MultiChannelTimeSampledSignalPlot.__init__(self, data=data, plottype=plottype,
                          plotparams=plotparams)


class EpochedMultiChannelTimeSampledSignalPlot(MultiChannelPlot):

    def __init__(self, emcss, mcshape=None, geometry=None, range2d=None, plotparams={}):
        '''
        We expect data to be a MultiChannelContinuousProcessEventSet
        '''

        plotparams['colormap'] = jet
        plotparams['border_visible'] = True

        self.__plotparams = plotparams # a dictionary of plot parameters

        if mcshape is None:
            mcshape = emcss.mcshape

        MultiChannelPlot.__init__(self, data=emcss, shape=mcshape,
                            geometry=geometry, range2d=range2d, xlabel='Time (s)', ylabel='Epoch')

        self.__zmapper = self._get_zmapper()

        zmapper = property(
        lambda self: (self.__zmapper), None, None,
        "")

    def _check_data(self, data):
        #assert isinstance(data, MultiChannelPointProcessEventSet)
        return data

    def _check_nchannels(self):
        return self.data.nchannels

    def _iter_datasource(self):
        if (self.geometry is not None ) and (
            self.data.channelnames is not None ):
            channels = self.geometry.mapfrom1Dto1D(
                array=numpy.arange(self.nchannels, dtype=numpy.int),
                pointnames = self.data.channelnames)
        else:
            channels = numpy.arange(self.nchannels)
        for i in channels:
            yield self.data[:,i,:].samples.get()

    def _get_xmapper(self):
        time = ArrayDataSource(self.data.samplingtimes())
        return LinearMapper(range=DataRange1D(time))

    def _get_ymapper(self):
        emcss = self.data
        eventnos = ArrayDataSource(numpy.arange(emcss.nepochs,0,-1))
        return LinearMapper(range=DataRange1D(eventnos), low=1, high=emcss.nepochs)

    def _get_zmapper(self):
        low = self.data.samples.get().min()
        high = self.data.samples.get().max()
        zr = ArrayDataSource(numpy.arange(low,high,10))
        return LinearMapper(range=DataRange1D(zr), low=low, high=high)

    def _get_plot(self, plotdata):
        #xmapper = self._get_xmapper()
        #_cmap = Trait(jet, Callable)
        #image_index.on_trait_change(self._metadata_changed,
        #                                  "metadata_changed")

        #image_value = ImageData(data=numpy.array([]), value_depth=1)
        #image_value_range = DataRange1D(image_value)
        #image_value.data = plotdata

        #plot = CMapImagePlot(index=image_index, value=image_value,
        #            index_mapper=GridMapper(range=image_index_range),
        #            value_mapper=_cmap(image_value_range))

        ##              index_mapper=GridMapper(range=image_index_range),
        pd = ArrayPlotData(imagedata=plotdata)
        #plot = ImagePlot(pd, index_mapper=GridMapper(range=image_index_range))
        plot = Plot(data=pd, padding=0)
        if len(self.plots) > 0:
                plot.range2d = self.plots[0].range2d
        #index_mapper=GridMapper(range=image_index_range))
        plot.img_plot("imagedata",
                      name="my_plot",
                      xbounds=self.data.samplingtimes(),
                      ybounds=numpy.arange(0,self.data.nepochs),
                      **self.__plotparams)
        # origin="top left",
        return plot

