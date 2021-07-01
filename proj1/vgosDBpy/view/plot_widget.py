from PySide2.QtWidgets import QTabWidget, QWidget, QVBoxLayout

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from vgosDBpy.data.plotFunction import Plotfunction_class
from vgosDBpy.model.data_axis import DataAxis
from vgosDBpy.view.plotlines import createLine2D, createSmoothCurve
from vgosDBpy.data.readNetCDF import not_S1, is_multdim_var_list
from vgosDBpy.view.AxesToolBox import AxesToolBox

class PlotFigure(FigureCanvas):
    '''
    View of matplotlib plots
    Can display several axes
    '''
    def __init__(self, figure = Figure(tight_layout = True), parent = None):

        '''
        figure [matplotlib.Figure]
        parent [QWidget]
        '''

        # Save figure and create Qt instance that displays plots from matplotlib
        self.figure = figure
        super(PlotFigure, self).__init__(self.figure)

        # Plot-related functions and variables
        self.plot_function = Plotfunction_class()
        self.timeInt = 1
        self.paths = []
        self.vars = []
        self.items = []
        self.Mult_item_added = False

        # Saves list of DataAxis belonging to the figure
        self._ax = []


        self.draw()



    def isPlotting(self):
        '''
        Checks if any DataAxis exists to this plot
        '''
        return len(self._ax) > 0

    ############### Getters & Setters

    def getDataAxis(self):
        '''
        Returns list of axes that belongs to the figure [list of DataAxis]
        '''
        return self._ax

    def getItems(self):
        '''
        Returns list of items that belongs to the figure [list of QNodeItems]
        '''
        return self.items

    def getFigure(self):
        '''
        Returns figure [matplotlib.Figure]
        '''
        return self.figure

    ############### Update graphics

    def updatePlot(self):
        '''
        Updates drawing of the plot
        '''
        self.figure.autofmt_xdate()
        self.draw()


    ################ Add/update axis and figure

    def addAxis(self, data_axis):
        '''
        Adds axis to the figure

        data_axis [DataAxis]
        '''
        axis = self.figure.add_axes(data_axis.getAxis())
        self._ax.append(data_axis)

    def removeAxis(self, data_axis):
        '''
        Removes axis from the figure

        data_axis [DataAxis]
        '''
        axis = data_axis.getAxis()
        self._ax.remove(data_axis)
        self._ax.clear()


    def exists(self, data_axis):
        '''
        Checks if the data_axis is in the figure

        data_axis [DataAxis]
        '''
        return data_axis in self._ax

    def updateFigure(self,items,timeUpdated=False,custom_x_axis=False,use_degrees=True):
        '''
        Updates figure with the given items

        items [list of QStandardItem]

        timeUpdated [boolean] decides whether this function is called when switching the indices between time / non-time
        '''
        # Checks if items exists
        if len(items) > 0:

            # Discards the old graph
            self.clearAxes()
            
            if timeUpdated is False:
                self.paths = []
                self.vars = []
                self.items = []

                for itm in items:
                    self.paths.append(itm.getPath())
                    self.vars.append(itm.labels)
                    self.items.append(itm)
            
            if not_S1(self.paths, self.vars):
                axis, data = self.plot_function.plotFunction(self.paths, self.vars, self.figure, self.timeInt, custom_x_axis=custom_x_axis,use_degrees=use_degrees)
                is_mult = is_multdim_var_list(self.paths, self.vars)

                if is_mult!= -1:
                    items.append(items[is_mult])
                # print("\nview.plot_widget.updateFigure()\nlen(axis):",len(axis))
                # print("len(data):",len(data))
                # print("len(items):",len(items),"\n")
                if(len(items) == 1):
                    for i in range(len(axis)):
                        self.addAxis(DataAxis(axis[i], data[i], items[0]))
                else:
                    for i in range(len(axis)):
                        self.addAxis(DataAxis(axis[i], data[i], items[i]))
                # Refresh canvas
                self.updatePlot()
            else:
                raise ValueError('Can not plot a string')


    def plot_script(self,paths,vars,figure, time):
        """
            Create a plot with the given data on the given figure
            input:
                paths: [array[strings]]
                vars: [array[strings]]
                figure: Figure
                time: int
        """
        self.clearAxes()
        axis, data = self.plot_function.plotFunction(paths, vars, figure, -1)
        for i in range(len(axis)):
            self.addAxis(DataAxis(axis[i], data[i], items[i]))
        self.updatePlot()

    def append_plot(self, item, custom_x_axis=False,use_degrees=True):
        """
        Appends existing figue with the given item
        item: [QStandardItem]
        custom_x_axis: [boolean] (optional argument)
        use_degrees: [boolean] (optional argument)
        """
        #add new item
        self.paths.append(item.getPath())
        self.vars.append(item.labels)
        self.items.append(item)
        self.clearAxes()
        if not_S1(self.paths,self.vars):
            axis, data = self.plot_function.plotFunction(self.paths, self.vars, self.figure, self.timeInt, custom_x_axis, use_degrees)
            for i in range(len(axis)):
                data_axis = DataAxis(axis[i], data[i], self.items[i])
                self.addAxis(data_axis)

            # Refresh canvas
            self.updatePlot()
        else:
            raise ValueError('Can not plot a string')


    # def x_axis_append_plot(self, item):
    #     """
    #     Changes existing x-axis to user selected data
    #     item: [QStandardItem]
    #     """
    #     #add new item
    #     #self.paths.append(item.getPath())
    #     #self.vars.append(item.labels)
    #     #self.items.append(item)
    #     self.clearAxes()
    #     if not_S1(self.paths,self.vars):
    #         axis, data = self.plot_function.plotFunction(self.paths, self.vars, self.figure, self.timeInt)
    #         for i in range(len(axis)):
    #             data_axis = DataAxis(axis[i], data[i], self.items[i])
    #             self.addAxis(data_axis)

    #         # Refresh canvas
    #         self.updatePlot()
    #     else:
    #         raise ValueError('Can not plot a string')


    def timeChanged(self):
        """
        Keeps track if the user changed the status of default plotting time on x-axis
        """
        self.updateFigure(self.items, timeUpdated = True)
        self.parentWidget().getPlotToolBar().updateDataAxis(self._ax)

    ###### Resetting methods

    def clearFigure(self):
        '''
        Reset the entire figure
        '''
        self.paths = []
        self.vars = []
        self.items = []
        self.Mult_item_added = False

        self.clearAxes()

    def clearAxes(self):
        '''
        Resets the list of DataAxis and removes them from the figure
        '''
        self.figure.clear()
        for ax in self._ax:
            self.removeAxis(ax)
        self._ax = []
        self.updatePlot()


    ######## Methods for saving figure as image

def saveCanvas(file_name, fig):
    '''
    Saves current figure as a pdf

    file_name [str]
    '''
    with PdfPages(file_name) as pdf:
        pdf.savefig(fig)


class PlotWidget(QWidget):
    '''
    Widget that brings the PlotFigure together with a navigation toolbar that allows different
    matplotlib features such as zoom and move in plot.
    '''
    def __init__(self, parent = None):
        '''
        parent [QWidget]
        '''
        super(PlotWidget, self).__init__(parent)

        self.plot_canvas = PlotFigure(parent = self)
        self.nav_toolbar = NavigationToolbar(self.plot_canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.nav_toolbar)
        layout.addWidget(self.plot_canvas)
        self.setLayout(layout)

    def getPlotToolBar(self):
        '''
        Get the AxesToolBox belonging to this instance
        '''
        return self.parentWidget().parentWidget().parentWidget().plot_toolbox

    def getDataAxis(self):
        '''
        Get the list of DataAxis active in this instance
        '''
        return self.plot_canvas.getDataAxis()

    def getItems(self):
        '''
        Get list of QStandardItems in this instance
        '''
        return self.plot_canvas.getItems()
