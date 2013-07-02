import numpy as np
from uts.core.uniformtime import isuts, ismcuts
from uts.processing.cstft import strft
from traits.api import HasTraits, Range, Bool, Float, Instance, \
     on_trait_change, Enum, Array, Button
from traitsui.api import View, Item, Group, ButtonEditor
from enable.component_editor import ComponentEditor
from chaco.api import ArrayPlotData, VPlotContainer, Plot, DataRange1D, \
     DataRange2D, jet, create_line_plot, OverlayPlotContainer
from chaco.tools.api import RangeSelection, LineInspector

eps = np.power(2.0,-52.0)

class STFTView(HasTraits):

    duration = Range(low='minduration',high='maxduration')
    starttime = Range(low='minstarttime',high='maxstarttime')
    vplots = Instance(VPlotContainer)
    automaxpower = Bool(True)
    maxpower = Range(low=0.,high=100.,value=30.0)
    dynamicrange = Range(low=0,high=100.,value=50.0)
    highfrequency = Range(low='minfrequency',high='maxfrequency')
    value_range = DataRange1D()
    stepduration = Range(low=0.,high=1.0,value=0.004)
    windowsize = Range(low=16,high=4096, value=1024)
    fftsize = Enum(1024,16,32,64,128,256,512,2048,4096)
    windowtype = Enum('gaussian','boxcar','hanning','hamming','blackman','bartlett')
    forward = Button(label='>')
    backward = Button(label='<')
    fforward = Button(label='>>')
    fbackward = Button(label='<<')
    _defaultstep = Range(low=0.1,high=1.0,value=0.25) # of duration
    _defaultwidth = Range(low=1,high=1920,value=900)
        
    traits_view = View(Item('vplots', editor=ComponentEditor(), show_label=False),
                       Item('automaxpower', show_label=True),
                       Item('maxpower', show_label=True),
                       Item('dynamicrange', show_label=True),
                       Item('highfrequency', show_label=True),
                       Item('duration', show_label=True),
                       Group(
                           Item('windowsize', show_label=True),
                           Item('fftsize', show_label=True),
                           Item('windowtype', show_label=True),
                           label='STFT',orientation= 'horizontal'),
                       Group(
                           Item('fbackward',show_label=False),
                           Item('backward',show_label=False),
                           Item('forward',show_label=False),
                           Item('fforward',show_label=False),
                           label='Controls',orientation= 'horizontal'),
                       width=1200, height=650, resizable=True, title="STFTView")

    def __init__(self, mcuts, starttime=None, duration=None):
        
        super(STFTView, self).__init__()

        if starttime is None:
            starttime = mcuts.timerange[0]
        if duration is None:
            duration = mcuts.duration
        if isuts(mcuts):
            mcuts = mcuts.to_multichannel()
        if not ismcuts(mcuts):
            raise TypeError(
                "mcuts should be a MultiChannelUniformTimeSeries object")


        self._renderers = []

        self.__mcuts = mcuts
        self.minduration = mcuts.dt
        self.maxduration = mcuts.duration
        self.duration = duration
        self.minfrequency = 0.
        self.maxfrequency = mcuts.fs/2.
        self.highfrequency = mcuts.fs/2.
        self.minstarttime = mcuts.timerange[0]
        self.maxstarttime = mcuts.timerange[1]
        #self.overview = RecViewNotes(recordingset)
        self.starttime = starttime
        
        self.__endtime = starttime + duration
        self.stepduration = duration / float(self._defaultwidth) # choose a reasonable setting
        self.on_trait_change(self._episode_changed, ['starttime','duration'])
        self.arrayplotdata = ArrayPlotData()
        self.on_trait_change(self._change_stftparams,
                        ['windowsize', 'fftsize', 'windowtype', 'stepduration'])
        self.__nchannels = mcuts.nchannels
        self.__specgramheight = 257
        self._plots = []
        
        self._containers = []
        self._set_sound()
        self.vplots = self._setup_vplots()
        

        #self.on_trait_change(self._set_sound, ['duration','starttime'])
    
    def _setup_vplots(self):
        vplots = VPlotContainer(stack_order='top_to_bottom', use_backbuffer=True)
        for i in range(self.__nchannels):
            container = OverlayPlotContainer(padding = 0, fill_padding = True,
                                     bgcolor = "lightgray", use_backbuffer=True)
            padding = [30,0,10,20]
            plot = Plot(self.arrayplotdata, padding=padding)
            self._plots.append(plot)
            container.add(plot)
            vplots.add(container)
            renderer = plot.img_plot("channel_%d"%i,
                           name="my_plot",
                           xbounds="samplingtimes_%d"%i,
                           ybounds="frequencies_%d"%i,
                           origin='bottom left',colormap=jet,
                           value_range=self.value_range)[0]
            renderer.index_range.set_bounds(low=(self.starttime,0.0),
                                        high=(self.__endtime,self.highfrequency))
            self._renderers.append(renderer)
            self._containers.append(container)
        return vplots

    def _set_sound(self):
        s = self.__mcuts[self.starttime: self.__endtime]
        if isuts(s):
            s = s.to_multichannel()
        self._signal = s
        self._set_spectrograms()
        
    
    def _change_stftparams(self, obj, name, old, new):
        # 'windowsize', 'fftsize', 'windowtype', 'stepduration'
        if self.fftsize < self.windowsize:
            warnings.warn("Windowsize cannot be larger than fftsize "
            "setting windowsize to fftsize", UserWarning)
            self.windowsize = self.fftsize
        self._set_spectrograms()
        
        
    def _set_spectrograms(self):
        starttime = self.starttime
        endtime = self.__endtime
        self._specgrams = []
        stepsize = int(np.round(self.stepduration*self._signal.fs))
        spgs = []
        maxpow = -np.inf
        for i in range(self._signal.nchannels):
            spgs.append(10*np.log10(strft(a=self._signal.samples[:,i], 
                        windowsize=self.windowsize, 
                        stepsize=stepsize,
                        windowtype=self.windowtype,
                        fftsize=self.fftsize,
                        power=True)/2e-7))
            curmax = spgs[i].max()
            if curmax > maxpow:
                maxpow = curmax
        self._spgmaxpower = maxpow
        if self.automaxpower:
            self.maxpower = maxpow
        self.value_range.low = self.maxpower - self.dynamicrange
        self.value_range.high = self.maxpower
        for i in range(self.__nchannels):
            self.arrayplotdata.set_data('channel_%d'%i,spgs[i])
            self.arrayplotdata.set_data('samplingtimes_%d'%i,np.linspace(starttime,endtime,spgs[i].shape[1]+1))
            self.arrayplotdata.set_data('frequencies_%d'%i,np.linspace(0,self._signal.fs/2.,spgs[i].shape[0]+1))

        
    
    def _episode_changed(self, obj, name, old, new):
        self.__endtime = self.starttime + self.duration
        self.stepduration = self.duration / float(self._defaultwidth)
        self._set_sound()
        
        
    def _automaxpower_changed(self, object, old, new):
        if new == True:
            self.maxpower = self._spgmaxpower
        for r in self._renderers:
            r.value_range.set_bounds(low=self.maxpower-self.dynamicrange,high=self.maxpower)
        r.request_redraw()
        
    
    def _maxpower_changed(self, old, new):
        if self.automaxpower == False:
            for r in self._renderers:
                r.value_range.set_bounds(low=new-self.dynamicrange,high=new)
            r.request_redraw()
            
            
    def _dynamicrange_changed(self, old, new):
        for r in self._renderers:
            r.value_range.low = self.maxpower - new
         # I don't know why but just updating one seems to do
        r.request_redraw()
            
    
    def _highfrequency_changed(self, old, new):
        for r in self._renderers:
            low = r.index_range.low
            high = r.index_range.high
            newlow = (low[0], 0.0)
            newhigh = (high[0], new)
            r.index_range.set_bounds(low=newlow,high=newhigh)
    
    def _forward_fired(self):
        self.starttime = self.starttime + self.duration * self._defaultstep
    def _backward_fired(self):
        self.starttime = self.starttime - self.duration * self._defaultstep
    def _fforward_fired(self):
        self.starttime = self.starttime + self.duration
    def _fbackward_fired(self):
        self.starttime = self.starttime - self.duration
       