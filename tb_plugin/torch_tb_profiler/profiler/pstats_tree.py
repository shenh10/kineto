import pstats
from collections import namedtuple, defaultdict
from typing import Dict, List, Any
import copy

PStatsFrame = namedtuple('pstats_frame', [
    'key',
    'func_name',
    'filepath',
    'nc',
    'cc',
    'tt',
    'time_per_call',
    'ct',
    'ct_ratio',
    'time_per_prim_call',
    'children'
])

def gen_pstats_tree(stats, topk=15) -> List[PStatsFrame]:
    node_id_book = {}
    key = 0
    def calc_callees(s):
        nonlocal key
        s.all_callees = all_callees = {}
        root_func = None
        for func, (cc, nc, tt, ct, callers) in s.stats.items():
            if not func in all_callees:
                all_callees[func] = {}
                node_id_book[func] = str(key)
                key += 1
            for func2, caller in callers.items():
                if not func2 in all_callees:
                    all_callees[func2] = {}
                    node_id_book[func2] = str(key)
                    key += 1
                all_callees[func2][func]  = caller
            if len(callers) == 0 and root_func is None:
                root_func = func
        return root_func
    path_id_book = {}
    path_key = 0
    """
    Use path key to identify each different path to root.
    
    Python callstack is formulated as a Directed Cyclic Graph.
    To traverse this graph and give a unique key for each entry of displayed table, we need to deal with:
        1) Infinity loop. The node which appears in current path again will be considered as a cycle start.
            we will cut this backward connection thus not appear in pstats table (
                instead, visulized graph image would show this connection)
                  ----> A ---> B ---> C ---> D               
                  |                          |      ->     A ---> B ---> C ---> D 
                  -------<----<-----<---------             
        2) Multipath: there might be multiple paths from root A to a node D. We need to assign differnt IDs to
           node D (PStatsFrame['key']) for frontend table row key in avoid of ambigiousity.
                   A --1-> B --2-> C --3-> D <---
                   |                            |
                   ------->---4->----->---------
    """
    def traverse(call_dict, func, cur_path):
        # sortby ct
        nonlocal path_key
        _dict = {k:v for k, v in sorted(call_dict[func].items(), key=lambda item: item[1][3], reverse=True)[:min(topk, len(call_dict[func]))]}
        for name, val in _dict.items():
            children = []
            if node_id_book[name] in set(cur_path):
                continue
            path_name = '->'.join(cur_path + [node_id_book[name]])
            cur_path.append(node_id_book[name])
            path_id_book[path_name] = (path_key, copy.deepcopy(cur_path))
            path_key += 1
            if len(call_dict[name]):
                children = traverse(call_dict, name, cur_path)
            yield PStatsFrame(
                path_id_book[path_name],
                name[2],
                name[0] + ':' + str(name[1]),
                val[0],
                val[1],
                pstats.f8(val[2]),
                pstats.f8(val[2]/val[0]),
                pstats.f8(val[3]),
                "{:.2f} %".format(100 * val[3]/stats.total_tt),
                pstats.f8(val[3]/val[1]),
                children
            )
            del cur_path[-1]

        
    columns = [
        {'name': 'Function', 'type': 'string', 'key': 'func_name'},
        {'name': 'Filename:Line Number', 'type': 'string', 'key': 'filepath'},
        {'name': 'Number of Calls', 'type': 'string', 'key': 'nc'},
        {'name': 'Primitive Calls', 'type': 'string', 'key': 'cc'},
        {'name': 'Self-function Total Time (s)', 'type': 'string', 'key': 'tt'},
        {'name': 'Time Per Call (s)', 'type': 'string', 'key': 'time_per_call'},
        {'name': 'Cumulative Time (s)', 'type': 'string', 'key': 'ct'},
        {'name': 'Cumulative Time Percentage', 'type': 'string', 'key': 'ct_ratio'},
        {'name': 'Time Per Primitive Call (s)', 'type': 'string', 'key': 'time_per_prim_call'},
    ]
    root_func = calc_callees(stats)
    if root_func is None:
        return []
    result = {}
    cur_path = [node_id_book[root_func]]
    data = sorted(list(traverse(stats.all_callees, root_func, cur_path)), key=lambda x: x.ct, reverse=True)
    return columns, data
