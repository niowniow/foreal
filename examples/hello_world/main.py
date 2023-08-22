import foreal

from .taskgraph import get_taskgraphs

# the taskgraphs is were all interesting things happen, see taskgraph.py
taskgraph = get_taskgraphs()

# we would like to compute the graph for the following request
request = {}

# start computation
foreal.compute(taskgraph, request)
