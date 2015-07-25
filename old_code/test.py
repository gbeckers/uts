from traits.api import HasTraits, Instance
from traitsui.api import View, Item
from chaco.api import Plot, ArrayPlotData, GridPlotContainer
from chaco.tools.api import LineInspector
from enable.component_editor import ComponentEditor

class TwoPlots(HasTraits):
    
    data = ArrayPlotData()
    plots = Instance(GridPlotContainer)
    traits_view = View(Item('plots', editor=ComponentEditor(), 
                        show_label=False), width=800, height=600,
                        resizable=True, title="Chaco Plot")
    
    def __init__(self, x, y1, y2):
        super(TwoPlots, self).__init__()
        self.set_data(x, y1, y2)      # populates ArrayPlotData
        self.plots = self.get_plots() # GridPlotContainer
        
    def set_data(self, x, y1, y2):
        "Adds numpy arrays to ArrayPlotData trait"
        self.data.set_data('x', x)
        self.data.set_data('y1', y1)
        self.data.set_data('y2', y2)
    
    def get_plots(self):
        "Returns GridPlotContainer containing plots"
        container = GridPlotContainer(shape=(2,1))
        for valuelabel in ('y1', 'y2'):
            container.add(self.get_subplot(valuelabel))
        return container
        
    def get_subplot(self, valuelabel):
        "Returns subplot"
        subplot = Plot(self.data)
        renderer = subplot.plot(('x', valuelabel))[0]
        li = LineInspector(renderer, axis="index", write_metadata=True,
                is_listener=True, color='magenta', is_interactive=True)
        renderer.overlays.append(li)
        return subplot
        
if __name__ == "__main__":
    import numpy
    x = numpy.linspace(0.0,1.0,50)
    y1 = numpy.sin(2*numpy.pi*x)
    y2 = numpy.cos(2*numpy.pi*x)
    TwoPlots(x,y1,y2).configure_traits()
    
        
        
