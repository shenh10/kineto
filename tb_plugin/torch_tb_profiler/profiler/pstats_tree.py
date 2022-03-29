import pstats
from collections import namedtuple, defaultdict
from typing import Dict, List, Any

PStatsFrame = namedtuple('pstats_frame', [
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
    def calc_callees(s):
        s.all_callees = all_callees = {}
        root_func = None
        for func, (cc, nc, tt, ct, callers) in s.stats.items():
            if not func in all_callees:
                all_callees[func] = {}
            for func2, caller in callers.items():
                if not func2 in all_callees:
                    all_callees[func2] = {}
                all_callees[func2][func]  = caller
            if len(callers) == 0 and root_func is None:
                print("hit ###### func", func)
                root_func = func
        return root_func

    visited = defaultdict(int)
    def traverse(call_dict, func):
        visited[func] += 1
        # sortby ct
        _dict = {k:v for k, v in sorted(call_dict[func].items(), key=lambda item: item[1][3], reverse=True)[:min(topk, len(call_dict[func]))]}
        for name, val in _dict.items():
            children = []
            if len(call_dict[name]) and ((name not in visited) or (visited[name] < 2)):
                children = traverse(call_dict, name)
            yield PStatsFrame(
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
    data = sorted(traverse(stats.all_callees, root_func), key=lambda x: x.ct, reverse=True)
    return columns, data
