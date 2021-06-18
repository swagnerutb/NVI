
from PySide2.QtWidgets import QTreeView, QTableView, QAbstractItemView
from PySide2.QtCore import QItemSelectionModel, Signal

from vgosDBpy.model.qtree import TreeModel
from vgosDBpy.model.table import VariableModel, DataModel

class QWrapper(QTreeView):
    '''
    Widget that inherits from QTreeView
    Visual representation of the wrapper as a folder structure

    Imports TreeModel which is the model representation of the wrapper
    '''

    def __init__(self, root_path, parent=None):
        '''
        root_path [string] is the path to the folder of the vgosDB session
        parent [QWidget]
        '''
        super(QWrapper, self).__init__(parent)

        # Setup model
        self._model = TreeModel(['Name'], root_path)
        self.setModel(self._model)

        # Selection of items
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.selection = self.selectionModel()

        # Expansion options
        self.expandToDepth(0)
        self.setExpandsOnDoubleClick(True)

        # Size-related
        self.resizeColumnToContents(0)

    def getModel(self):
        '''
        Returns TreeModel
        '''
        return self._model

    def getWrapper(self):
        '''
        Returns Wrapper belonging to the TreeModel
        '''
        return self._model.getWrapper()

class VariableTable(QTableView):
    '''
    Widget that inherits from QTableView
    Visual representation of the items in a table

    Imports VariableModel
    '''

    def __init__(self, parent = None, header = ['Name', 'Unit', 'Dimension', 'Dtype']):
        '''
        parent [QWidget]
        header [list of strings] is the labels for the table, the number of strings has to equals
        the number of columns filled with data
        '''

        super(VariableTable, self).__init__(parent)

        # Setup model
        self._model = VariableModel(header, parent)
        self.setModel(self._model)

        # Selection of items
        #self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.selection = self.selectionModel()



        # Size
        '''
        max_width = self.frameSize().width()
        self.setMaximumWidth(max_width)
        self.setColumnWidth(0, 6/10*max_width)
        self.setColumnWidth(1, 4/10*max_width)
        '''

    def getModel(self):
        '''
        Returns VariableModel
        '''
        return self._model

    def updateVariables(self, var_list):
        '''
        Updates content of table model

        var_list [list of QStandardItems]
        '''
        self._model.updateVariables(var_list)

        # Updates size of column when content is changed
        for i in range(self._model.columnCount()):
            self.resizeColumnToContents(i)

class DataTable(QTableView):
    '''
    Displays data from DataModel which has values from a variable together with timestamp
    '''

    # SIGNAL used to detect mouse release (The ordinary signal in Qt was not working)
    custom_mouse_release = Signal()

    def __init__(self, parent = None):
        '''
        parent [QWidget] is the parent widget
        '''
        super(DataTable, self).__init__(parent)

        # Setup model
        self._model = DataModel('', parent) # just use the two functions get_name_to_print and get_unit_to_print istead of 'Value'
        self.setModel(self._model)

        # Selection of items
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.selection = self.selectionModel()

    ######### Getters & setters

    def getModel(self):
        '''
        Returns DataModel
        '''
        return self._model

    def resetModel(self):
        '''
        Resets the model's instance variables
        '''
        self._model.resetModel(reset_header = True)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.selection = self.selectionModel()

    ######### Event listener

    def mouseReleaseEvent(self, e):
        '''
        Overwriting the ordinary mouse release event with another emitted signal
        '''
        super(DataTable, self).mouseReleaseEvent(e)
        self.custom_mouse_release.emit()

    ########### Appearance-related methods

    def updateColumnSize(self):
        '''
        Updates size of column adjusted after the content
        '''
        for i in range(self._model.columnCount()):
            self.resizeColumnToContents(i)

    ############### Update methods

    def updateData(self, items):
        '''
        Updates the data in the table

        items [list of QNodeItems] contains the nodes which points to the netCDF variables
        that should be displayed
        '''
        # Updates model
        self._model.updateData(items)

        # Updates size of column when content is changed
        self.updateColumnSize()

    def appendData(self,items):
        '''
        Updates the data in the table

        items [list of QNodeItems] contains the nodes which points to the netCDF variables
        that should be displayed
        '''

        # Updates model
        self._model.appendData(items)

        # Updates size of column when content is changed
        self.updateColumnSize()

    ######### Update through DataAxis methods

    def updateFromDataAxis(self, data_axis):
        '''
        Updates the table by giving it the data_axis, this gives a one to one correspondance with
        the plot

        data_axis [list of DataAxis] is the DataAxis instances which the table should display
        '''

        # Update model
        self._model.updateFromDataAxis(data_axis)

        # Updates size of column when content is changed
        self.updateColumnSize()


    def updateMarkedRows(self, data_axis):
        '''
        Mark selected data from plot in the table

        data_axis [list of DataAxis] is the rows that should be updated
        '''
        self.selection.clear()

        for ax in data_axis:
            col_index = self._model.dataaxis_to_column_map.get(ax)

            if col_index == None:
                continue

            selected_data = ax.getMarkedData()

            for row_index in selected_data:
                model_item = self._model.item(row_index, col_index)
                item_index = self._model.indexFromItem(model_item)
                self.selection.select(item_index, QItemSelectionModel.Select) # This line takes a lot of time to execute
