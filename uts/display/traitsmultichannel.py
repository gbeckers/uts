import numpy

from traits.api import HasTraits, Instance, Range, Array, \
     on_trait_change, Float, Bool, Trait, Callable, DelegatesTo, Dict

from traits.trait_handlers import TraitDictObject
from traitsui.api import View, Item, Group

from chaco.api import ArrayDataSource, DataRange1D, LinePlot, \
     FilledLinePlot, BarPlot, ScatterPlot, LinearMapper, add_default_grids, \
     PlotAxis, GridPlotContainer, HPlotContainer, ArrayPlotData, \
     create_scatter_plot, ArrayPlotData, jet, Plot, GridDataSource, DataRange2D,\
     GridMapper, ImageData, ImagePlot, OverlayPlotContainer, DataLabel
from chaco.tools.api import PanTool, ZoomTool, BroadcasterTool, \
     LineInspector, PointMarker
from chaco.tools.cursor_tool import CursorTool1D, BaseCursorTool
import chaco.default_colormaps
from enable.api import Window, KeySpec
from enable.component_editor import ComponentEditor
from enable.api import BaseTool
from enable.events import KeyEvent

from uts3 import MultiChannelUniformTimeSeries

class MultiChannelPlot(HasTraits):

    plot = Instance(GridPlotContainer)
    xrange = Instance(DataRange1D)
    line_index = Float
    right_key = Instance(KeySpec, args=("Right",))
    right_plus_ctrl_key = Instance(KeySpec, args=("Right","control"))
    right_plus_alt_key = Instance(KeySpec, args=("Right","alt"))
    left_key = Instance(KeySpec, args=("Left",))
    left_plus_ctrl_key = Instance(KeySpec, args=("Left","control"))
    left_plus_alt_key = Instance(KeySpec, args=("Left","alt"))


    traits_view = View(Item('plot', editor=ComponentEditor(), show_label=False),
                       width=1200, height=800, resizable=True,
                       title="Chaco Plot")

    def __init__ (self, geometry, labels=None,
                  ftitle='Multi-Channel View', xlabel='', ylabel='', *args, **kw):

        super(MultiChannelPlot, self).__init__()

        self.__nchannels = self._check_nchannels()

        self.__geometry = geometry
        self.__shape = (self.__geometry.ny, self.__geometry.nx)
        self.__xlabel = xlabel
        self.__ylabel = ylabel
        self.__labels = self._check_labels(labels)

        self.__plots = []
        self.overlay_components = None
        self.tool_components = None
        self.plot = self._get_plots()
        self.keytool = KeyTool(self.plot)
        self.keytool.on_trait_change(self.key_pressed, 'keyevent')
        self.plot.tools.append(self.keytool)

        
    nchannels = property(
        lambda self: (self.__nchannels), None, None,
        "number of channels")
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



    def _check_data(self, data):
        pass

    def _check_nchannels(self, data):
        pass

    def key_pressed(self, event):
        dt = None
        if self.right_key.match(event):
            dt = self.data.dt
        elif self.right_plus_alt_key.match(event):
            dt = self.data.dt * 3
        elif self.right_plus_ctrl_key.match(event):
            dt = self.data.dt * 5
        elif self.left_key.match(event):
            dt = -self.data.dt
        elif self.left_plus_alt_key.match(event):
            dt = -self.data.dt * 3
        elif self.left_plus_ctrl_key.match(event):
            dt = -self.data.dt * 5
        if dt:
            self.plots[0].components[0].index.metadata['selections'] = self.line_index + dt
            self.plots[0].request_redraw()

    def _check_labels(self, labels):
        if labels is None:
            return None
        l = {}
        for key, item in labels.items():
            l[key] = (ArrayDataSource(numpy.array(item)),
                      ArrayDataSource(numpy.ones(len(item))))
        return l

    def _get_plot(self):
        pass

    def _get_plots(self):
        # padding (l,r,t,b)
        container = GridPlotContainer(shape=self.shape, spacing=(2.0,2.0),
                                      padding=(50,10,20,40),  bgcolor = "white",#"lightgray",
                                      use_backbuffer=True)

        for i in range(self.nchannels):

            #ol =  OverlayPlotContainer(padding=0, fill_padding = True,
            #                           use_backbuffer=True)
            p,r = self._get_plot(i, overlay_components=self.overlay_components,
                                  tool_components=self.tool_components)
            if not self.plots: # this is the first plot, let's see if they have overlay components
                self.overlay_components = p.components[0].overlays
                self.tool_components = p.components[0].tools
                p.components[0].index.on_trait_change(self.inspector_moved,'metadata[]')

            # labels
            colors = ['blue','red', 'green', 'black', 'magenta', 'cyan', 'yellow']

            if self.labels is not None:
                signal = ArrayDataSource(numpy.linspace(0.0,1.2,10))
                ymapper = LinearMapper(range=DataRange1D(signal))
                for i, (key, (labeltimes, zeros)) in enumerate(self.labels.items()):
                    signal = ArrayDataSource(numpy.linspace(0.0,1.0,10))
                    ymapper = LinearMapper(range=DataRange1D(signal))
                    sp = ScatterPlot(index = labeltimes, value = zeros,
                                     index_mapper = LinearMapper(range=self.xrange),
                                     value_mapper = ymapper,
                                     color=colors[i], marker='inverted_triangle', line_width=0.0,
                                     marker_size=8)
                    p.add(sp)
            container.add(p)
            self.__plots.append(p)

        return container

    def inspector_moved(self):
        metadata = self.plots[0].components[0].index.metadata
        if metadata is not None and "selections" in metadata:
            self.line_index = metadata["selections"][0]
        else:
            self.line_index = 0.

    def draw_pdf(self, filename, size=[600,800],
                 dest_box=(0.1, 0.1, -0.1, -0.1), units='inch'):

        from chaco.pdf_graphics_context import PdfPlotGraphicsContext
        container = self.plot # CHECK THIS
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

        MultiChannelPlot.__init__(self, data=mcppes)

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

        MultiChannelPlot.__init__(self, data=mcppes, geometry=geometry)

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


## commented out 20120521
# class ViewRangePlot(HasTraits):

#     def __init__(starttime, endtime):
#         self.plot = LinePlot(x=[starttime,endtime], y=[0.0,0.0])



class MultiChannelUniformTimeSeriesPlot(MultiChannelPlot):

    _viewstart = Float
    _viewend = Float

    def __init__(self, data, plottype='lineplot', geometry=None,
                 plotparams={}, labels=None, starttime=None, endtime=None, **kwargs):
        '''
        Parameters
        ----------
        data: a MultiChannelUniformTimeSeries
        labels: a dictionary with keys indicating color, and values arrays of time values 
        '''

        self.arrayplotdata = ArrayPlotData()
        self.xrange = DataRange1D()
        self.on_trait_change(self.update_xrange, 'xrange')
        self.li = []

        self.__data = data
        if starttime is None:
            starttime = -data.origintime
        if endtime is None:
            if self.__data.ntimesamples > 50000:
                endtime = float(self.__data.index_to_time(50000))
            else:
                endtime = starttime + data.duration
            
        self.viewstart = float(starttime)       
        self.viewend = float(endtime)
        self.on_trait_change(self.set_array, ['viewstart','viewend'])
        
        self.set_array()

        plotparams['border_visible'] = True
        self.__plotparams = plotparams # a dictionary of plot parameters

        MultiChannelPlot.__init__(self, geometry=geometry,
                                  labels=labels, xlabel='Time (s)', 
                                  ylabel='Amplitude')

    data = property(
        lambda self: (self.__data), None, None,
        "multi-channel data")

    yrange = property(
        lambda self: (self.__yrange), None, None,
        "")
    
    def _check_data(self, data):

        #assert isinstance(data, tsig.MultiChannelUniformTimeSeries)
        #print type(data)
        return data

    def _check_nchannels(self):
        return self.data.nchannels

    #def _get_xrange(self, samplingtimes):
        #return DataRange1D(ArrayDataSource(self.__data.samplingtimes()))
    
    def _get_yrange(self, samples):
        samples = self.__data.samples.get(copy=False)
        signal = ArrayDataSource(numpy.linspace(samples.min(),samples.max(),10))
        return DataRange1D(signal)

    def update_xrange(self):
        for plot in self.plots:
            plot.index_range = self.xrange
    
    def set_array(self):
        #FIXME this may return a different object (i.e. when we have a 1-channel signal)
        sel = self.data[self.viewstart:self.viewend].transpose(['time','channel'])
        for i in range(self.data.nchannels):
            self.arrayplotdata.set_data('samples_%d'%i,sel[:,i].samples.get(copy=False))
        samplingtimes = sel.samplingtimes()
        self.arrayplotdata.set_data('samplingtimes', samplingtimes)
        self.__yrange = self._get_yrange(sel.samples.get(copy=False))
        
    def _get_plot(self, i, overlay_components=None, tool_components=None):
        if (not (self.data.geometry is None)) and (not (self.data.channelnames is None)):
            j = self.data.geometry.mapfrom1Dto1D(numpy.arange(self.nchannels),
                                                 self.data.channelnames)[i]
        else:
            j = i
        plot = Plot(self.arrayplotdata, padding=[0,0,0,0])
        renderer = plot.plot(('samplingtimes', 'samples_%d'%j), name='samples_%d'%j,
                      **self.__plotparams)[0]
        #if len(self.plots) > 0:
        plot.value_range = self.yrange #self.plots[0].value_range
        plot.index_range = self.xrange #self.plots[0].index_range
            #plot.x_mapper.range = self.xrange #self.plots[0].value_range
            #plot.y_mapper.range = self.yrange #self.plots[0].index_range
        if i == 0:
            self.index = renderer.index # trick, see if this is still necessary
        else:
            renderer.index = self.index
        if (i % self.shape[1]) == 0:
            plot.y_axis.visible = True
        else:
            plot.y_axis.visible = False
        if i >= self.nchannels - self.shape[1]:
            plot.x_axis.visible = True
        else:
            plot.x_axis.visible = False
        if overlay_components is None:
            li = LineInspector(renderer, axis="index",
                                               write_metadata=True,
                                               is_listener=True, color='magenta',
                                               is_interactive=True,
                                               metadata_name='selections')
            renderer.overlays.append(li)
            renderer.overlays.append(ZoomTool(renderer, drag_button="right",
                                          always_on=True,
                                          tool_mode="box",
                                          axis="index",
                                          enable_wheel=True))
            #csr = CursorTool1D(renderer, drag_button="left",
            #                      color='blue')
            #   renderer.overlays.append(csr)
        else:
            for c in overlay_components:
                renderer.overlays.append(c)
        if tool_components is None:
            renderer.tools.append(PanTool(renderer, constrain=False,
                                  constrain_direction="x"))
            #renderer.tools.append(KeyTool(renderer))
        else:
            for c in tool_components:
                renderer.tools.append(c)
        
        return plot,renderer

class EpochedMultiChannelUniformTimeSeriesPlot(MultiChannelPlot):

    samplingtimes = Array
    cursor1 = Instance(BaseCursorTool)
    master_value_range = DataRange1D()
    autovrange = Bool(True)
    vrangemin = Float()
    vrangemax = Float()
    vrangelow = Float()
    vrangehigh = Float()

    traits_view = View(Group(
        Item('plot', editor=ComponentEditor(), show_label=False)
        ),
                       width=1200, height=800, resizable=True,
                       title="Chaco Plot")

    def __init__(self, emcss, geometry=None, zmapper=None, plotparams={}):
        '''
        We expect data to be a MultiChannelContinuousProcessEventSet
        '''

        self.__data = emcss

        self.arrayplotdata = ArrayPlotData()
        
        if zmapper is None:
            self.__zmapper = self._get_zmapper()
        else:
            self.__zmapper = zmapper
        self.set_array()

        plotparams['colormap'] = jet
        plotparams['border_visible'] = True

        self.__plotparams = plotparams # a dictionary of plot parameters

        self._renderers = {}
        self.__cursors = []
        self._value_ranges = {}

        MultiChannelPlot.__init__(self, geometry=geometry,
                                  xlabel='Time (s)', ylabel='Epoch')

    data = property(
        lambda self: (self.__data), None, None,
        "multi-channel data")
    zmapper = property(
        lambda self: (self.__zmapper), None, None,
        "")


    def _check_data(self, data):
        #assert isinstance(data, MultiChannelPointProcessEventSet)
        return data

    def _check_nchannels(self):
        return self.data.nchannels

    def _get_zmapper(self):
        low = float(self.data.samples.get().min())
        high = float(self.data.samples.get().max())
        self.vrangelow = low
        self.vrangehigh = high
        self.vrangemin = low
        self.vrangemax = high
        zr = ArrayDataSource(numpy.arange(low,high,2))
        #zr = ArrayDataSource(self.data.samples.get())
        return LinearMapper(range=DataRange1D(zr), low=low, high=high)

    def set_array(self):
        for i in range(self.data.nchannels):
            self.arrayplotdata.set_data('samples_%d'%i,
                    self.data[:,i,:].samples.transpose(['epoch','time']).get(copy=False))
        samplingtimes = self.data.samplingboundarytimes()
        self.arrayplotdata.set_data('samplingtimes', samplingtimes)

    def _get_plot(self, i, overlay_components=None, tool_components=None):
        if (not (self.data.geometry is None)) and (not (self.data.channelnames is None)):
            j = self.data.geometry.mapfrom1Dto1D(numpy.arange(self.nchannels),
                                                 self.data.channelnames)[i]
        else:
            j = i
        plot = Plot(self.arrayplotdata, padding=[0,0,0,0])

        if len(self.plots) > 0:
            plot.range2d = self.plots[0].range2d
        #print self.arrayplotdata["samples_%d"%j].shape
        #print self.arrayplotdata["samplingtimes"].shape
        
        renderer = plot.img_plot("samples_%d"%j,
                                name="my_plot",
                                xbounds="samplingtimes",
                                ybounds=numpy.arange(0,self.data.nepochs+1),
                                origin='top left',
                                **self.__plotparams)[0]
        self._renderers[j] = renderer
        self._value_ranges[j]= renderer.value_range
        
        if i == 0:
            self.index = renderer.index
        else:
            renderer.index = self.index
        if (i % self.shape[1]) == 0:
            plot.y_axis.visible = True
        else:
            plot.y_axis.visible = False
        if i >= self.nchannels - self.shape[1]:
            plot.x_axis.visible = True
        else:
            plot.x_axis.visible = False

        lix = LineInspector(renderer, axis="index_x",
                                           write_metadata=True,
                                           is_listener=True, color='white',
                                           is_interactive=True)
        renderer.overlays.append(lix)
        liy = LineInspector(renderer, axis="index_y",
                                           write_metadata=True,
                                           is_listener=True, color='white',
                                           is_interactive=True)
        renderer.overlays.append(liy)
        renderer.tools.append(PanTool(renderer, constrain=False,
                                  constrain_direction="x", drag_button="middle"))
        renderer.overlays.append(ZoomTool(renderer, drag_button="right",
                                      always_on=True,
                                      tool_mode="box",
                                      axis="index",
                                      enable_wheel=True))
        return plot,renderer
    
    def _vrangelow_changed(self, object, old, new):
        print "vrangelow changed"
        if self.autovrange == False:
            print "autovrange false"
            for i,r in self._renderers.items():
                print new
                r.value_range.set_bounds(low=new,high=self.vrangehigh)
                r.request_redraw()
                print "redraw %d" % i
                
    def _vrangehigh_changed(self, object, old, new):
        print "vrangehigh changed"
        if self.autovrange == False:
            print "autovrange false"
            for i,r in self._renderers.items():
                r.value_range.set_bounds(low=self.vrangelow,high=new)
                print new
                r.request_redraw()
                print "redraw %d" % i
            
        
    def _autovrange_changed(self, object, old, new):
        if new == False:
            low = self.vrangelow
            high = self.vrangehigh
            for i,r in self._renderers.items():
                r.value_range.set_bounds(low=low,high=high)
                r.request_redraw()
        else:
            for i,r in self._renderers.items():
                low = self.arrayplotdata["samples_%d"%i].min()
                high = self.arrayplotdata["samples_%d"%i].max()
                r.value_range.set_bounds(low=low,high=high)
                r.request_redraw()
        

class MultiChannelImagePlot(HasTraits):
    
    pd = Instance(ArrayPlotData, ())
    plot = Instance(Plot)
    time = Float
    #right_zoom_key = Instance(KeySpec, args=(">",))

    traits_view = View(
            Item('plot',
                 editor=ComponentEditor(),
                 show_label=False,
            ),
            width=360,
            height=360,
            resizable=True
        )

    def __init__(self, mcuts, clip=None, mcutsp=None, 
                 colormap='jet', inversecolor=False, time=0.0, **kwargs):

        super(MultiChannelImagePlot, self).__init__()
        self.mcutsp = mcutsp
        self._mcuts = mcuts
        self._mappingindex = mcuts.geometry.indexfrom1Dto2D(mcuts.channelnames)
        self._imar = numpy.zeros(mcuts.geometry.shape[::-1], dtype=mcuts.dtype)
        self.inversecolor = inversecolor

        self.on_trait_change(self.update_data, 'time')

        self.update_data()
  
        # Create the plot object, set some options, and add some tools
        plot = self.plot = Plot(self.pd, default_origin="top left")
        plot.img_plot('imagedata',
                      name='Spectrogram',
                      xbounds=(0, mcuts.geometry.dx*(mcuts.geometry.nx-1)),
                      ybounds=(0, mcuts.geometry.dy*(mcuts.geometry.ny-1)),
                      colormap=chaco.default_colormaps.color_map_name_dict[colormap]
                      )
        plot.x_axis.orientation = "top"
        plot.padding = 50
        plot.padding_top = 75
        plot.tools.append(PanTool(plot))
        zoom = ZoomTool(component=plot, tool_mode="box", always_on=False)
        plot.overlays.append(zoom)

        range_obj = plot.plots['Spectrogram'][0].value_mapper.range
        if clip is None:
            range_obj.high = 0.5e-3
            range_obj.low = -0.5e-3
        else:
            range_obj.high = clip[1]
            range_obj.low = clip[0]

        if not (mcutsp is None):
            self.on_trait_change(self.time_changed, 'mcutsp.line_index')

    def time_changed(self):
        if self.mcutsp and (self._mcuts.timerange[0] <= self.mcutsp.line_index <= self._mcuts.timerange[1]):
            self.time = self.mcutsp.line_index

    def update_data(self):
        if self.mcutsp and (self._mcuts.timerange[0] <= self.mcutsp.line_index <= self._mcuts.timerange[1]):
            timeindex = self._mcuts.time_to_index(float(self.time))
            #optimize this by asking samples directly instead of mc object
            if self.inversecolor:
                self._imar[self._mappingindex] = -self._mcuts[timeindex].samples[:].flatten()
            else:
                self._imar[self._mappingindex] = self._mcuts[timeindex].samples[:].flatten()
        else:
            self._imar[self._mappingindex] = numpy.zeros(self._mcuts.nchannels)
        self.pd.set_data('imagedata', self._imar)


class KeyTool(BaseTool):

    keyevent = Instance(KeyEvent)

    def normal_key_pressed(self, event):
        #print "KeyTool", type(event), event
        self.keyevent = event
        
    #def normal_mouse_move(self, event):
    #    print "Screen point:", event.x, event.y