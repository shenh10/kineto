from collections import namedtuple
from typing import Dict, List, Any



PARAMS_CUR_RANK = 'params_cur_rank'
NUM_FORWARD_MACS_CUR_RANK = 'num_fwd_macs_cur_rank'
NUM_FORWARD_FLOPS_CUR_RANK = 'num_fwd_flops_cur_rank'
FORWARD_FLOPS_CUR_RANK = 'fwd_FLOPS_cur_rank'
FORWARD_LATENCY = 'fwd_latency'
LAYER_DETAILS = 'layer_details'
INPUT_SHAPE = 'input_shape'


ModelStats = namedtuple('Stats', [
    'key',
    'name',
    'type',
    'params',
    'params_percentage',
    'macs',
    'macs_percentage',
    'flops',
    'duration',
    'latency_percentage',
    'extra_repr',
    'children'])


def parse_model_stats(json_dict: Dict[str, Any]):
    detail = []
    overviews = []
    overviews.append({'title': 'Monitor Iteration', 'value': json_dict[PARAMS_CUR_RANK]})
    overviews.append({'title': 'Total Number of Params Current Rank', 'value': json_dict[PARAMS_CUR_RANK]})
    overviews.append({'title': 'Total Number of Forward MACs Current Rank', 'value': json_dict[NUM_FORWARD_MACS_CUR_RANK]})
    overviews.append({'title': 'Total Number of Forward FLOPs Current Rank', 'value': json_dict[NUM_FORWARD_FLOPS_CUR_RANK]})
    overviews.append({'title': 'Total Forward FLOPs Per Second Current Rank', 'value': json_dict[FORWARD_FLOPS_CUR_RANK]})
    overviews.append({'title': 'Total Forward Latency', 'value': json_dict[FORWARD_LATENCY]})

    
    if LAYER_DETAILS in json_dict:
        detail = _process_model_statistics(json_dict[LAYER_DETAILS])
    return overviews, detail

def _process_model_statistics(
        modules_nodes: Dict[str, Any]) -> List[ModelStats]:

    root_name = None
    for key, val in modules_nodes.items():
        name, _type, _id = key.split(':')
        if _id == '0':
            root_name = key
            break
    if root_name is None:
        return []
    
    def process_modules(h_modules: List[str]):
        for name in h_modules:
            m = modules_nodes[name]
            child_stats = []
            if len(m['children']) > 0:
                child_stats = list(process_modules(m['children']))
            yield ModelStats(
                m['unique_id'],
                m['name'],
                m['class_name'],
                m['detail']['params'],
                m['detail']['params_percentage'],
                m['detail']['macs'],
                m['detail']['macs_percentage'],
                m['detail']['flops'],
                m['detail']['duration'],
                m['detail']['latency'],
                str(m['extra_repr']),
                child_stats)

    data = sorted(process_modules([root_name]), key=lambda x: x.name)
    return data
