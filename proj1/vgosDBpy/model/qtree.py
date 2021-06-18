import importlib.util
import os
from PySide2.QtGui import QStandardItemModel, QStandardItem
from PySide2 import QtCore

from vgosDBpy.wrapper.parser import Parser
from vgosDBpy.wrapper.tree import Node

class TreeModel(QStandardItemModel):
    '''
    Model that represents the wrapper
    It is inherited from QStandardModel which can be used to implement a tree structure
    Is used together with QTreeView to generate a visual representation
    '''

    def __init__(self, header, root_path, parent=None):
        '''
        root_path [string]
        wrapper_name [string]
        parent [QWidget]
        '''
        super(TreeModel,self).__init__(parent)

        # Is set up in setupModel
        self._wrapper = None

        self.setupModel(root_path)
        self.setHorizontalHeaderLabels(header)


    def flags(self, index):
        '''
        Let us choose if the selected items should be enabled, editable, etc

        index [QModelIndex]
        '''

        if not index.isValid():
            return 0

        return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable #| QtCore.Qt.ItemIsEditable # Uncomment if you want to be able to edit it

    def getWrapper(self):
        '''
        Returns the tree structure that represents the parsed wrapper [Wrapper]
        '''
        return self._wrapper

    # Model setup
    def setupModel(self, root_path):
        '''
        Parsing the wrapper
        (Imports Parser class)

        root_path [string]
        '''
        parser = Parser(root_path)
        parser.parseWrapper()
        root_parent = parser.getWrapperRoot()
        self.recursive(root_parent, self.invisibleRootItem())

        self._wrapper = parser.getWrapper()

    def recursive(self, node, parent):
        '''
        Recursively read the wrapper.Wrapper class and load it to this tree structure

        node [wrapper.tree.Node]
        parent [wrapper.tree.Node]

        '''
        if node.isNetCDF():
            item = NetCDFItem(node)
        else:
            item = QNodeItem(node)
        parent.appendRow(item)
        if node.hasChildren():
            children = node.getChildren()
            for row in range(node.getChildCount()):
                c = children[row]
                self.recursive(c, item)
        else:
            if node.isNetCDF() is False and not node.isHistFile():
                item.appendRow(QNodeItem('Empty'))


class QNodeItem(QStandardItem):
    '''
    Custom data item for the QStandardItemModel
    '''
    _type = 1110


    def __init__(self, node):
        '''
        node [wrapper.tree.Node]
        '''

        super(QNodeItem, self).__init__(0,2)
        self.labels = str(node)
        self.node = node

    ####### Getters ################

    def getPath(self):
        '''
        Returns path of node
        '''
        return self.node.getPath()

    def getNode(self):
        return self.node

    ####### Type checking ###########

    def isNetCDF(self):
        '''
        Checks if the node points to a netCDF
        '''
        return self.node.isNetCDF()

    def isHistFile(self):
        '''
        Checks if the node points to a .hist file
        '''
        return self.node.isHistFile()

    def type(self):
        '''
        Returns type
        '''
        return QNodeItem._type

    ###### Data reading and setting

    def data(self, role = QtCore.Qt.DisplayRole):
        '''
        Decides how to display data in the Qt view
        '''
        if role == QtCore.Qt.DisplayRole:
            return self.labels

        elif role == QtCore.Qt.EditRole:
            return self.node

        else:
            return None

    def setData(self, data, role = QtCore.Qt.EditRole):
        '''
        Decides how to edit data in the Qt view
        '''
        if role == QtCore.Qt.EditRole:
            self.labels = data

        elif role == QtCore.Qt.DisplayRole: # Do not really do anything?
            self.labels = data

        else:
            return False

        self.emitDataChanged()
        return True

    ###### __ __ methods #########

    def __str__(self):
        return self.labels

    def __repr__(self):
        return self.labels

    def __hash__(self):
        return hash(self.labels)*33 + hash(self.labels)*22 + hash(str(self.node.parent))*11


    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.node == other.node
        else:
            return False



class NetCDFItem(QNodeItem):

    '''
    Inherited from QNodeItem

    This should be used to differ QNodeItem which points to netCDF files
    '''

    _type = 1111

    def __init__(self, node):
        '''
        node [wrapper.tree.Node]
        '''
        super(NetCDFItem, self).__init__(node)

    def type(self):
        NetCDFItem._type



class Variable(QNodeItem):

    '''
    Inherited from QNodeItem

    self.node becomes the NetCDFItem that the variable belongs to
    This should be used to differ variables that are stored in this format which points to netCDF files
    '''

    _type = 1112

    def __init__(self, variable_name, node):
        '''
        variable_name [string]
        node [wrapper.tree.Node]
        '''
        super(Variable, self).__init__(node)
        self.labels = variable_name

    ########## Type checking ##################

    def type(self):
        NetCDFItem._type


class DataValue(QNodeItem):

    '''
    Inherited from QNodeItem

    This should be used to store single data values from the node
    '''

    _type = 1113


    def __init__(self, value, node = None, signal = None):
        '''
        value [Object] is the value that is belong to this cell/node
        node [wrapper.tree.Node]
        signal [QtCore.Signal] is necessary for editing the DataTable
        '''
        super(DataValue, self).__init__(node)
        self.value = value

        self.signal = signal
        self.custom_signal_bool = (signal != None)

    ########## Type checking ##################

    def type(self):
        NetCDFItem._type

    ############# Data reading and setting methods #######
    def data(self, role = QtCore.Qt.DisplayRole):
        '''
        Decides how to display data in the Qt view
        '''

        if role == QtCore.Qt.DisplayRole:
            return str(self.value)
        elif role == QtCore.Qt.EditRole:
            return self.value
        else:
            return None

    def setData(self, data, role = QtCore.Qt.EditRole):
        '''
        Decides how to edit data in the Qt view

        Emits custom signal if one was given
        '''

        if role == QtCore.Qt.EditRole:
            self.value = data

        elif role == QtCore.Qt.DisplayRole: # Do not really do anything?
            self.value = data

        else:
            return False

        if self.custom_signal_bool is True:
            self.signal.emit(self.column())
        else:
            self.emitDataChanged()

        return True

    ######## __ __ methods ############
    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)
