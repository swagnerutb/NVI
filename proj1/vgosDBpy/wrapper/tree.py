import re
import os


class Wrapper:
    '''
    Class for representing a wrapper in a tree data structure
    '''

    # Pre-defined scopes will have the folder made automatically
    scopes = ['Session','Scan', 'Observation','Station']

    def __init__(self, wrapper_path):
        '''
        Constructor

        wrapper_path [string] is the path to the wrapper file
        '''

        # Path to wrapper file
        self.wrapper_path = wrapper_path

        # Get session name
        path = wrapper_path.split('/')
        self.session_name = path[-2]

        # Removes last which is the name of the wrapper file and get path to session folder
        path.pop()
        self.root_path = '/'.join(path)

        # Create root node
        self.root = Node(self.session_name, None, self.root_path)

        # Other instance variables
        self.pointer_map = {} # Keep track of pointers with a map
        self.hist_file = [] # Keeps track of the .hist files that the wrapper points to


    def addNode(self, name, parent = None, type = 'netCDF'):
        '''
        Add node in the tree, may be a file, data array or a folder/node

        name [string] is the node's name
        parent [string] is the name of the parent, if None then
        it have root as parent
        type [string] is the type of node, currently there exist 'node', 'netCDF' or 'hist'
        '''

        # Checks for parent
        if parent == None:
            parent_node = self.root
        else:
            parent_node = self.getNode(parent)

        path = self.generatePath(name, parent_node)

        # Checks the type of Node to be created
        if type == 'node':

            if not parent_node.childNodeExists(name):
                new_node = Node(name, parent_node, path)
            else:
                new_node = None

        elif type == 'netCDF':

            new_node = NetCDF_File(name, parent_node, path)

        elif type == 'hist':

            new_node = HistFile(name, parent_node, path)
            self.hist_file.append(new_node)

        else:

            raise InvalidArgument('Invalid type in', type)

        # Adds node to tree
        if new_node != None:

            parent_node.addChildNode(new_node)
            self.addPointer(new_node)

    def addPointer(self, node):
        '''
        Set new pointers

        node [Node] is the node to point to
        '''
        self.pointer_map[node.getName()] = node

    def getNode(self, name):
        '''
        NOTE: Will not work if there exists several files with the same name.

        Get a node by using the map of pointers

        name [string] is the name of the desired node
        '''
        return self.pointer_map[name]

    def getRoot(self):
        '''
        Returns the root node [Node]
        '''
        return self.root

    def getHistFile(self):
        '''
        Returns list of the history files [list of HistFile]
        '''
        return self.hist_file

    def getHistory(hist_path):
        '''
        Returns history file in text format [string]

        hist_path [string] is the path to the history file
        '''
        if os.path.isfile(hist_path):
            text = ''
            with open(hist_path) as file:
                for line in file:
                    text = text + line
            return text
        else:
            raise ValueError('Invalid path, does not exists:', hist_path)

    def inScope(name):
        '''
        Checks if a name is defined in the scopes

        name [string]

        Returns boolean
        '''
        return name.lower() in Wrapper.scopes

    def generatePath(self, name, parent):
        '''
        Return file path [string] to a desired node with the given name

        name [string]
        parent [Node]
        '''
        if name == self.session_name:
            node_path = self.root_path
        else:
            path = name
            while parent is not self.root:
                path = str(parent) + '/' + path
                node = parent
                parent = node.getParent()
            node_path = self.root_path + '/' + path
        last_folder = node_path.split('/')[-2].strip()
        if not last_folder in Wrapper.scopes:
            scope_string = []
            for s in Wrapper.scopes:
                scope_string.append('/' + s + '/')
                #
            node_path = re.sub(r'|'.join(scope_string), '/', node_path)
            node_path = re.sub(r'(/)\1*', '/', node_path)
        return node_path

    def __str__(self):
        indent = " " * 4
        s = ''
        for child in self.root.getChildren():
            line = str(child) + '\n'
            s = s + line
            if type(child) == Node:
                for sub_child in child.getChildren():
                    s = s + indent + str(sub_child) + '\n'
                    if type(sub_child) == Node:
                        for subsub_child in sub_child.getChildren():
                            s = s + indent*2 + str(subsub_child) + '\n'
        return s


### Tree structure
class Node(object):
    '''
    Defining a Node class for the tree data structure
    '''

    def __init__(self, name, parent, path):
        '''
        name [string]
        parent [string] to the node
        path [string] to the file which the Node represents
        '''
        self.name = name
        self.children = []
        self.parent = parent
        self.path = path


        self.netCDF = False
        self.hist = False


    def addChildNode(self, node):
        '''
        Add another node as a child to the current node

        node [Node]
        '''
        if self.childNodeExists(node) == False:
            self.children.append(node)


    def getParent(self):
        '''
        Returns parent to the current node [Node]
        '''
        return self.parent

    def getChildren(self):
        '''
        Returns list of children to the current node [list of Nodes]
        '''
        return self.children

    def getChildNode(self, name):
        '''
        Returns a child node with the desired name

        name [string]

        Raises error if name not found
        '''
        for obj in self.getChildren():
            if obj.compareName(name):
                return obj
        raise AttributeError(name + ' was not found in the ' + str(self) + ' folder')

    def getChildCount(self):
        '''
        Returns number of children to the Node [int]

        '''
        return len(self.children)

    def getName(self):
        '''
        Return name of current node [string]
        '''
        return self.name

    def getPath(self):
        '''
        Return path belonging to the Node [string]
        '''
        return self.path

    def hasChildren(self):
        '''
        Checks if the Node has children

        Returns boolean
        '''
        if self.children == None:
            return False
        else:
            return len(self.children)>0

    def childNodeExists(self, name):
        '''
        Returns True if a childnode with the desired name exists
        Otherwise False

        name [string]
        '''
        for obj in self.getChildren():
            if obj.compareName(name):
                return True
        return False

    def compareName(self, name):
        '''
        Checks if a given name is equal to the name of the current node

        name [string]
        '''
        return self.name == str(name)

    def isNetCDF(self):
        '''
        Checks if the Node is a netCDF

        Used to seperate subclasses
        '''
        return self.netCDF

    def isHistFile(self):
        '''
        Checks if the Node is a history file

        Used to seperate subclasses
        '''
        return self.hist

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

class NetCDF_File(Node):
    '''
    Subclass to Node which should point to netCDF files in the tree structure
    '''

    def __init__(self, name, parent, path):
        '''
        name [string]
        parent [string] to the node
        path [string] to the file which the Node represents
        '''
        super().__init__(name, parent, path)
        self.children = None
        self.netCDF = True

    def addChildNode(self, obj):
        '''
        Overwrite Node's method to make sure that a netCDF file can not have child nodes
        '''
        raise TypeError('Tried assigning files to another file, needs to be a folder.')

class HistFile(Node):
    '''
    Subclass to Node which should point to history files in the tree structure
    '''
    def __init__(self, name, parent, path):
        '''
        name [string]
        parent [string] to the node
        path [string] to the file which the Node represents
        '''
        super().__init__(name, parent, path)
        self.children = None
        self.hist = True

    def addChildNode(self, obj):
        '''
        Overwrite Node's method to make sure that a history file can not have child nodes
        '''
        raise TypeError('Tried assigning files to another file, needs to be a folder.')

class PointerMap():
    '''
    NOT USED

    Hash map to keep track of pointers to nodes in the Tree
    Each key to the map will be defined by a node's name and it's parent.
    '''
    def __init__(self):
        self._map = {}

    def addPointer(self, node, parent):
        '''
        node [Node]
        parent [Node]
        '''
        self._map[PointerMap.generateKey(node,parent)] = node

    def removePointer(self, node, parent):
        '''
        node [Node]
        parent [Node]
        '''
        del self._map[generateKey(node, parent)]

    def getPointer(self, node_name, parent_name):
        '''
        node_name [string] is the name of the node
        parent_name [string] is the name of the parent

        Returns the node [Node]
        '''
        return self._map[PointerMap.getKey(node_name, parent_name)]

    def getKey(node_name, parent_name):
        '''
        node_name [string] is the name of the node
        parent_name [string] is the name of the parent

        Returns key [string]
        '''
        return node_name + '_' + parent_name

    def generateKey(node, parent):
        '''
        node [Node]
        parent [Node]

        Returns key [string]
        '''
        return getKey(node.getName(), parent.getName())
