import os
import inquirer

def ask_user_wrp(path):
    """
    User selection of path to .wrp file
    """
    dir_cont = ['### return ###']
    dir_cont_wrp = []
    dir_cont_dir = []
    for content in os.listdir(path):
        if(content.endswith('.wrp')):
            dir_cont_wrp.append(content)
        elif(os.path.isdir(os.path.join(path,content))):
            dir_cont_dir.append(content)

    dir_cont.extend(dir_cont_wrp) #Always get .wrp files first
    dir_cont.extend(dir_cont_dir)
    
    if(len(dir_cont) == 1):
        print('#####\nYou have entered a directory with no subdirectories or .wrp files. Please select another directory.\n#####')
        return ask_user_wrp(os.path.dirname(path))
    
    question = [inquirer.List('select',message='Please select a .wrp file or directory', choices=dir_cont)]
    answer = inquirer.prompt(question)['select']
    
    if(answer == '### return ###'):
        return ask_user_wrp(os.path.dirname(path))
    elif(answer.endswith('.wrp')):
        return os.path.join(path,answer)
    elif(os.path.isdir(os.path.join(path,answer))):
        return ask_user_wrp(os.path.join(path,answer))

def ask_user_g():
    """
    If user wishes to run the graphical interface.
    """
    questions = [inquirer.List('y_n',message='Would you like to open the graphical interface?', choices=["yes", "no"])]
    return inquirer.prompt(questions)['y_n']

path = os.environ.get('VGOSDB_DIR')

wrp_dir = ask_user_wrp(path)

graph_flag = ask_user_g()

os_cmd = 'python -m vgosDBpy ' + wrp_dir
os_cmd = 'python3 -m vgosDBpy ' + wrp_dir

if(graph_flag == "yes"):
    os_cmd = os_cmd+" -g"


os.system(os_cmd)