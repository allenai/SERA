import os
import glob
import logging
import json

from typing import List

##############################
# codebase folder wildcards

def convert_path_to_wildcard(path: str):
    assert "/" in path
    path, filetype = os.path.splitext(path)
    split_path = path.split("/")
    for i, component in enumerate(split_path):
        if i > 0:
            split_path[i] = "*"
    return "/".join(split_path) + filetype

def get_folder_wildcards(path: str, filetype_filter=".py"):
    wildcard_set = set()
    for dirpath, dirnames, filenames in os.walk(path):
        # For each folder, it gives us a list of filenames
        for filename in filenames:
            # We use os.path.join() to create the full, absolute path
            absolute_path = os.path.join(dirpath, filename)
            if not filetype_filter or absolute_path.endswith(filetype_filter):
                wildcard_set.add(convert_path_to_wildcard(absolute_path))
    return list(wildcard_set)

def find_code_folders(repo_path: str, repo_last_name: str, base_commit: str, top_level_folder: List[str]):
    cwd = os.getcwd()
    try:
        os.chdir(repo_path)
        # Get top level code folder
        code_path = None
        if top_level_folder:
            for tlf in top_level_folder:
                if os.path.exists(tlf):
                    code_path = tlf
        else:
            print("Trying to automatically find top level code folder...")
            if os.path.exists(repo_last_name):
                code_path = repo_last_name
            elif os.path.exists(repo_last_name.lower()):
                code_path = repo_last_name.lower()
            elif os.path.exists(os.path.join("src", repo_last_name)):
                code_path = os.path.join("src", repo_last_name)
            elif os.path.exists("src"):
                code_path = "src"
            else:
                print("Failed to find top level code folder, quitting now...")
        if code_path:
            wildcards = get_folder_wildcards(code_path)
        else:
            wildcards = []
    except Exception as e:
        raise
    finally:
        os.chdir(cwd)
    return wildcards

##############################
# code2flow

def split_function_path(func_path):
    """
    Expects in the form of lib/matplotlib/inset::InsetIndicator.set_alpha.
    Returns the file path and the function path. Here it would be lib/matplotlib/inset.py
    and InsetIndicator.set_alpha.
    """
    file_path, fn = func_path.split("::")
    module_name = os.path.split(file_path)[-1]
    if module_name.endswith(".py"):
        module_name = module_name[:-3]

    if not file_path.endswith(".py"):
        file_path += ".py"
    return file_path, fn

def get_full_path(folders, func, code_dict):
    file_name, func_name = split_function_path(func)
    found_file_path = ""
    found_times = 0
    # Search for the file containing the function
    func_components = func_name.split(".")
    for folder in folders:
        glob_path = os.path.join(folder, file_name)
        matches = [n for n in glob.glob(glob_path) if os.path.isfile(n)]
        for file_path in matches:
            if file_path in code_dict:
                file_text = code_dict[file_path]
            else:
                with open(file_path, "r") as f:
                    file_text = f.read()
                code_dict[file_path] = file_text
            if len(func_components) == 2 and func_components[0] in file_text and func_components[1] in file_text:
                found_file_path = file_path
                found_times += 1
            elif len(func_components) == 1 and func_components[0] in file_text:
                found_file_path = file_path
                found_times += 1 
                
    # Only return successfully if we found a unique occurence of the function
    if found_times == 1:
        return found_file_path + "::" + func_name
    else:
        return None

def convert_to_file_path(call_graph, folders, node_id_to_name, nodes):
    func_to_path_map = {}
    line_mappings = {}
    code_dict = {}
    # Reverse the node_id_to_name dictionary to get the ID of any name
    node_name_to_id = {}
    for k, v in node_id_to_name.items():
        node_name_to_id[v] = k

    # Get line mappings of each function file path and create map of func name to func path
    remove_nodes = set()
    for i, func in enumerate(call_graph.keys()):
        file_path = get_full_path(folders, func, code_dict)
        if file_path is None:
            remove_nodes.add(func)
        else:
            func_to_path_map[func] = file_path
            line_mappings[file_path] = int(nodes[node_name_to_id[func]]["label"].split(":")[0])

    new_call_graph = {}
    for func, neighbors in call_graph.items():
        # Skip removed nodes
        if func in remove_nodes:
            continue
        
        # Filter neighbors and map to paths in one step
        valid_neighbors = [
            func_to_path_map[nbr] 
            for nbr in neighbors 
            if nbr not in remove_nodes
        ]
        
        new_call_graph[func_to_path_map[func]] = valid_neighbors
    return new_call_graph, line_mappings

def convert_code2flow_to_adj(loaded_json):
    new_call_graph = {}
    id_to_name = {}
    nodes = loaded_json["nodes"]
    edges = loaded_json["edges"]
    # Fill in adjacency list
    for uid in nodes:
        new_call_graph[nodes[uid]["name"]] = set()
        id_to_name[uid] = nodes[uid]["name"]
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        new_call_graph[id_to_name[source]].add(id_to_name[target])
    # Convert sets to lists
    for node_name in new_call_graph:
        new_call_graph[node_name] = list(new_call_graph[node_name])
    return new_call_graph, id_to_name, nodes

def get_adj_list(repo_path: str, 
                 repo_last_name: str,
                 base_commit: str,
                 relevant_folders: List[str], 
                 metadata_dir: str, overwrite=False):
    cwd = os.getcwd()
    try:
        os.chdir(repo_path)
        cg_save_path = os.path.join(metadata_dir, f"{repo_last_name}_{base_commit[:5]}.json")
        if not os.path.exists(cg_save_path) or overwrite:
            # Generate call graph
            file_input = " ".join(relevant_folders)
            cmd = ["code2flow"] + [file_input] + ["--o", cg_save_path] + ["--quiet"]
            os.system(" ".join(cmd))
        with open(cg_save_path, "r") as f:
            call_graph = json.load(f)
        # Get call graph in adjacency list format
        adj_list, node_id_to_name, nodes = convert_code2flow_to_adj(call_graph["graph"])
        # Convert call graph to use full file paths, by default code2flow uses file_name::func_name
        adj_list, _ = convert_to_file_path(adj_list, [os.path.split(p)[0] for p in relevant_folders], node_id_to_name, nodes)
    except FileNotFoundError:
        # Sometimes codeflow fails, and then there is no file to read.
        logging.error(f"\t\tFailed: {repo_last_name}, repo_path: {repo_path}, cur_dir: {os.getcwd()}")
        adj_list = None
    except Exception as e:
        print(e)
        adj_list = None
    finally:
        os.chdir(cwd)
    return adj_list