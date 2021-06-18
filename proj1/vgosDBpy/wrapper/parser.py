from os import getcwd
import sys
import importlib
from vgosDBpy.wrapper. tree import Wrapper

class Parser:
    '''
    Class for parsing wrapper files (*.wrp) in vgosDB format
    '''

    # Constructor
    def __init__(self, root_path):
        '''
        root_path [string] is the path to the wrapper

        '''
        self.wrapper = Wrapper(root_path)
        self._active_scope = []
        self.path_to_wrp = root_path

    ##################################################
    # Methods which keep track of the current scope in the wrapper_path
    # Represented as a queue which keeps the most recent mentioned scope
    # Scopes are defined in wrapper.py
    
    def get_wrp_path(self):
        return self.path_to_wrp

    def getWrapper(self):
        return self.wrapper

    def getActiveScope(self):
        if len(self._active_scope) == 0:
            return None
        else:
            for i in range(len(self._active_scope)):
                if Wrapper.inScope(self._active_scope[-1-i]):
                    return self._active_scope[-1-i]
            return None

    def addScope(self, scope):
        self._active_scope.append(scope)

    def removeScope(self, scope):
        self._active_scope.remove(scope)

    ##################################################


    def getWrapperRoot(self):
        return self.wrapper.getRoot()

    """
    Method is called by 'createNewWrp' to get a list of all directories in old wrp
    """
    def find_all_directories(self,path):
        directories = []
        with open(path,'r') as scr:
            for line in scr:
                l = line.strip()
                if l.lower().startswith('default_dir'):
                    directories.append(l.split()[1] )
        return directories



    def parseWrapper(self):
        '''
        Methods that parses the wrapper files which contains information and
        pointers to relevant files in one VLBI session
        '''
        # Define current folder, None if the wrapper has no default_dir
        active_folder = None

        # Open file
        with open(self.path_to_wrp,'r') as src:

            # Loop through each file
            for line in src:

                # Correct format of line
                line = line.strip()

                # First keyword
                first_keyword = line.split()[0].lower()

                # Skip comments in wrapper
                if first_keyword == '!':
                    continue

                # Check for beginning of sections
                elif first_keyword == 'begin':
                    keyword = line.split()[1]
                    self.addScope(keyword)

                # Check for end of sections
                elif first_keyword == 'end':
                    keyword = line.split()[1]
                    active_folder == None
                    self.removeScope(keyword)

                # Check for setting the default_dir (active_folder)
                elif first_keyword == 'default_dir':
                    active_folder = line.split()[1]
                    self.wrapper.addNode(active_folder, self.getActiveScope(), 'node')

                # Checks if line is giving a netCDF pointer
                elif line.endswith('.nc'):
                    file_name = line.split()[-1]
                    if active_folder == None:
                        parent_folder = self._active_scope[-1]
                    else:
                        parent_folder = active_folder
                    self.wrapper.addNode(file_name, parent_folder, 'netCDF')

                elif line.endswith('.hist'):
                    file_name = line.split()[-1]
                    self.wrapper.addNode(file_name, active_folder, 'hist')

                else:
                    pass
                    # Else statement for debugging purposes
        return self.wrapper
