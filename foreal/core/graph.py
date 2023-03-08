"""
Copyright 2022 Swiss Federal Institute of Technology (ETH Zurich), Matthias Meyer

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

import copy
import traceback
from copy import copy, deepcopy
from hashlib import new
from sqlite3 import InternalError

import dask
import numpy as np
from dask.base import tokenize
from dask.core import flatten, get_dependencies
from dask.delayed import Delayed
from dask.optimization import cull, fuse, inline, inline_functions

from foreal.config import get_setting, setting_exists
from foreal.convenience import NestedFrozenDict, dict_update, base_name, KEY_SEP


class NodeFailedException(Exception):
    def __init__(self, exception=None):
        """The default exception when a node's forward function fails and failsafe mode
        is enable, i.e. the global setting `fail_mode` is not set to `fail`.
        this exception is caught by the foreal processing system and depending on the
        global variable `fail_mode`, leads to process interruption or continuation.

        Args:
            exception (any, optional): The reason why it failed, e.g. another exception.
                Defaults to None.
        """
        if get_setting("fail_mode") == "warning" or get_setting("fail_mode") == "warn":
            print(exception)
        self.exception = exception

    def __str__(self):
        return str(self.exception)


class FailSafeWrapper:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        try:
            return self.func(*args, **kwargs)
        except Exception as ex:
            trace = traceback.format_exc(2)
            return NodeFailedException(trace)


def it(x, fail_mode=None):
    """Helper function to transform input callable into
    dask.delayed object, i.e. internal representation for foreal.
    "foreal it!"

    Args:
        x (callable): Any input callable which is supported by dask.delayed

        fail_mode (str or None, optional): Instance specific option for fail-safe mode.
            None will use the system default setting. Defaults to None.

    Returns:
        dask.delayed: The output can be used in foreal task graphs
    """
    if fail_mode is None:
        if setting_exists("fail_mode"):
            fail_mode = get_setting("fail_mode")
        else:
            fail_mode = "fail"
    if fail_mode != "fail":
        return dask.delayed(FailSafeWrapper(x))
    else:
        return dask.delayed(x)


def update_key_in_config(request, old_key, new_key):
    if "config" in request:
        if "keys" in request["config"]:
            if old_key in request["config"]["keys"]:
                request["config"]["keys"][new_key] = request["config"]["keys"].pop(
                    old_key
                )


class Node(object):
    def __init__(self, **kwargs):
        self.config = locals().copy()
        # FIXME: This works but there are better solutions!
        while "kwargs" in self.config:
            if "kwargs" not in self.config["kwargs"]:
                self.config.update(self.config["kwargs"])
                break
            self.config.update(self.config["kwargs"])

        del self.config["kwargs"]
        del self.config["self"]

    def merge_config(self, request):
        """Each request contains configuration which may apply to different
        node instances. This function collects all information that apply to _this_
        node (including it's preset configs) and adds a `self` keyword to the request.

        Args:
            request (dict): The request and configuration options.

        Returns:
            dict: A new request which specific to this node.
        """
        protected_keys = ["self", "config"]

        new_request = {}
        new_request.update(request)
        new_request["self"] = {}
        new_request["self"].update(deepcopy(self.config))
        if "indexers" in request:
            dict_update(
                new_request["self"], {"indexers": deepcopy(request["indexers"])}
            )

        if "config" in request:
            PROTECTED_CONFIG_KEYS = ["global", "types", "keys"]

            # new_request['config'] = {}

            # We'll go through all parameters in the request's config
            # and add them to the self parameters

            # We assume that anything within 'config' is global
            for key in request["config"]:
                if key in PROTECTED_CONFIG_KEYS:
                    continue
                new_request["self"][key] = request["config"][key]

                # # add to new_request
                # new_request['config'][key] = request["config"][key]

            # add specific global config entries
            if "global" in request["config"]:
                for key in request["config"]["global"]:
                    new_request["self"][key] = request["config"]["global"][key]

                # # add to new_request
                # new_request['config']['global'] = request["config"]["global"]

            # add type specific configs (overwrites global config)
            if "types" in request["config"]:
                if type(self).__name__ in request["config"]["types"]:
                    dict_update(
                        new_request["self"],
                        request["config"]["types"][type(self).__name__],
                    )

                # # add to new_request
                # new_request['config']['types'] = request["config"]["types"]

            # add key specific configs (overwrites global and type config)
            if "keys" in request["config"]:
                if self.dask_key_name in request["config"]["keys"]:
                    dict_update(
                        new_request["self"],
                        request["config"]["keys"][self.dask_key_name],
                    )

                # TODO: It should be save to remove these keys from the new_request!?
                # new_request['config']['keys'] = {k:v for k,v in request["config"]["keys"].items() if k != self.dask_key_name}

                # # add to new_request
                # new_request['config']['keys'] = request["config"]["keys"]

        if "config" in request:
            new_request["config"] = request["config"]
        return new_request

    def configure(self, requests):
        """Before a task graph is executed each node is configured.
            The request is propagated from the end to the beginning
            of the DAG and each nodes "configure" routine is called.
            The request can be updated to reflect additional requirements,
            The return value gets passed to predecessors.

            Essentially the following question must be answered within the
            nodes configure function:
            What do I need to fulfil the request of my successor? Either the node
            can provide what is required or the request is passed through to
            predecessors in hope they can fulfil the request.

            Here, you must not configure the internal parameters of the
            Node otherwise it would not be thread-safe. You can however
            introduce a new key 'requires_request' in the request being
            returned. This request will then be passed as an argument
            to the __call__ function.

            Best practice is to configure the Node on initialization with
            runtime independent configurations and define all runtime
            dependant configurations here.

        Args:
            requests {List} -- List of requests (i.e. dictionaries).


        Returns:
            dict -- The (updated) request. If updated, modifications
                    must be made on a copy of the input. The return value
                    must be a dictionary.
                    If multiple requests are input to this function they
                    must be merged.
                    If nothing needs to be requested an empty dictionary
                    can be return. This removes all dependencies of this
                    node from the task graph.

        """
        if not isinstance(requests, list):
            raise RuntimeError("Please provide a **list** of request")
        if len(requests) > 1:
            raise RuntimeError(
                "Default configuration function cannot handle "
                f"multiple requests. "
                "Please provide a custom "
                f"configuration implementation. "
                f"{len(requests)} requests were provided.: {requests}"
            )

        request = requests[0]

        merged_request = self.merge_config(request)

        # set default
        merged_request["requires_request"] = True
        return merged_request

    def __call__(
        self, data=None, request=None, delayed=None, dask_key_name=None, fail_mode=None
    ):
        if dask_key_name is not None and KEY_SEP in dask_key_name:
            raise RuntimeError(
                f"Do not use a `{KEY_SEP}` character in your {dask_key_name=}"
            )

        if request is None:
            request = {}
        internal_request = self.merge_config(request)

        if delayed is None:
            delayed = get_setting("use_delayed")

        if fail_mode is None:
            if setting_exists("fail_mode"):
                fail_mode = get_setting("fail_mode")
            else:
                fail_mode = "fail"

        if fail_mode != "fail":
            forward_func = self.fail_safe_forward
        else:
            forward_func = self.forward

        if delayed:
            func = dask.delayed(forward_func)(
                data, internal_request, dask_key_name=dask_key_name
            )
            self.dask_key_name = func.key
            return func
        else:
            return forward_func(data=data, request=internal_request)

    def fail_safe_forward(self, *args, **kwargs):
        try:
            return self.forward(*args, **kwargs)
        except Exception as ex:
            trace = traceback.format_exc(2)
            return NodeFailedException(trace)

    def forward(self, data, request):
        raise NotImplementedError

    # def get_config(self):
    #     """returns a dictionary of configurations to recreate the state"""
    #     # TODO:
    #     raise NotImplementedError()

def generate_clone_key(current_node_name,to_clone_key,clone_id):
    return base_name(to_clone_key) + KEY_SEP + tokenize([current_node_name, to_clone_key, "foreal_clone", clone_id])

# @profile
def configuration(
    delayed,
    request,
    keys=None,
    default_merge=None,
    optimize_graph=True,
    dependants=None,
    clone_instead_merge=True,
):
    """Configures each node of the graph by propagating the request from outputs
    to inputs. Each node checks if it can fulfil the request and what it needs to fulfil
    the request. If a node requires additional configurations to fulfil the request it
    can set the 'requires_request' flag in the returned request and this function will
    add the return request as a a new input to the node's __call__().
    See also Node.configure()

    Args:
        delayed (dask.delayed or list): Delayed object or list of delayed objects
        request (dict or list): request (dict), list of requests
        keys (_type_, optional): _description_. Defaults to None.
        default_merge (_type_, optional): _description_. Defaults to None.
        optimize_graph (bool, optional): _description_. Defaults to True.
        dependants (_type_, optional): _description_. Defaults to None.

    Raises:
        RuntimeError: If graph cannot be configured

    Returns:
        dask.delayed: The configured graph
    """

    if not isinstance(delayed, list):
        collections = [delayed]
    else:
        collections = delayed

    # dsk = dask.base.collections_to_dsk(collections)
    dsk, dsk_keys = dask.base._extract_graph_and_keys(collections)
    if dependants is None:
        _, dependants = dask.core.get_deps(dsk)

    # dsk_dict = {k:dsk[k] for k in dsk.get_all_external_keys()}
    dsk_dict = {k: dsk[k] for k in dsk.keys()}

    if keys is None:
        keys = dsk_keys
    if not isinstance(keys, (list, set)):
        keys = [keys]

    work = list(set(flatten(keys)))
    # create a deepcopy, otherwise we might overwrite requests and falsify its usage outside of this function
    # request = deepcopy(request)
    if isinstance(request, list):
        # request = [NestedFrozenDict(r) for r in request if r]
        request = [r for r in request if r]
        if len(request) != len(work):
            raise RuntimeError(
                "When passing multiple request items "
                "The number of request items must be same "
                "as the number of keys"
            )

        # For each output node different request has been provided
        requests = {work[i]: [request[i]] for i in range(len(request))}
    else:
        # request = NestedFrozenDict(request)
        # Every output node receives the same request
        requests = {k: [request] for k in work}

    remove = {k: False for k in work}
    input_requests = {}
    # We will create a new graph with the configured nodes of the old graph
    # out_keys keeps track of the keys we have configured and
    # remember them for assembling the new graph
    out_keys = []
    seen = set()

    # using dict here for performance because we are doing `if key in work` later
    # (and not using sets because for unknown reasons it doesn't work)
    work = {k: True for k in work}

    while work:
        # new_work = []
        new_work = dict()

        out_keys += work
        deps = []
        for k in work:
            # if k not in requests:
            #     # there wasn't any request stored use initial config
            #     requests[k] = [config]

            # check if we have collected all dependencies so far
            # we will come back to this node another time
            # TODO: make a better check for the case when dependants[k] is a set, also: why is it a set in the first place..?
            if (
                k in dependants
                and len(dependants[k]) != len(requests[k])
                and not isinstance(dependants[k], set)
            ):
                continue

            if k not in requests:
                InternalError(f"Failed to find request for node {k}")

            # set configuration for this node k
            # If we create a delayed object from a class, `self` will be dsk_dict[k][1]
            argument_is_node = None
            if isinstance(dsk_dict[k], tuple):
                # check the first argument
                # # TODO: make tests for all use cases and then remove for-loop
                for ain in range(1):
                    if hasattr(dsk_dict[k][ain], "__self__"):
                        if isinstance(dsk_dict[k][ain].__self__, Node):
                            argument_is_node = ain
            # Check if we get a node of type Node class
            if argument_is_node is not None:
                # Yes, we've got a node class so we can use it's configure function
                # current_requests = [NestedFrozenDict(r) for r in requests[k] if r]
                current_requests = [r for r in requests[k] if r]
                new_request = dsk_dict[k][argument_is_node].__self__.configure(
                    current_requests
                )  # Call the class configuration function
                if not isinstance(new_request, list):
                    new_request = [new_request]
                # convert back to dicts (here we are allowed to modify it)
                new_request = [dict(r) for r in new_request]
            else:
                # We didn't get a Node class so there is no
                # custom configuration function: pass through or use default_merge
                if len(requests[k]) > 1:
                    if callable(default_merge):
                        # current_requests = deepcopy(requests[k])
                        current_requests = [r for r in requests[k] if r]
                        # current_requests = [NestedFrozenDict(r) for r in requests[k] if r]
                        new_request = default_merge(current_requests)
                        if not isinstance(new_request, list):
                            new_request = [new_request]
                        # convert back to dicts (here we are allowed to modify it)
                        new_request = [dict(r) for r in new_request]
                    else:
                        # try to merge if all requests are the same just take the first
                        first_hash = tokenize(requests[k][0])
                        for r in requests[k]:
                            if first_hash != tokenize(r):
                                raise RuntimeError(
                                    "No valid default merger supplied. Cannot merge requests. "
                                    "Either convert your function to a class Node or provide "
                                    f"a default merger. Failed requests are: {requests[k]}"
                                )
                        # all requests are the same, take the first one
                        new_request = requests[k][:1]
                else:
                    new_request = []
                    for r in requests[k]:
                        if r:
                            # sanitize request
                            rn = dict(r)
                            rn.pop("requires_request", None)
                            rn.pop("insert_predecessor", None)
                            rn.pop("clone_dependencies", None)
                            rn.pop("remove_dependency", None)
                            rn.pop("remove_dependenies", None)
                            new_request.append(rn)

                    # new_request = [dict(r) for r in requests[k] if r]

            # update dependencies
            # we're going to get all dependencies of this node
            # then, we'll check if this node requires to clone it's input path
            # If so, each cloned path gets a different request from this node
            # (these are contained in a list in `clone_dependencies`)
            # we are going to introduce new keys and new nodes in the graph
            # therefore we must update this nodes input keys (hacking it from/to dsk_dict[k][1])
            # for each clone
            insert_predecessor = []
            if (
                new_request[0] is not None
                and "insert_predecessor" in new_request[0]
                and new_request[0]["insert_predecessor"]
            ):
                insert_predecessor = new_request[0]["insert_predecessor"]
                del new_request[0]["insert_predecessor"]

            current_deps = get_dependencies(dsk_dict, k, as_list=True)
            k_in_keys = None
            if len(dsk_dict[k]) > 1:
                k_in_keys = deepcopy(dsk_dict[k][1])  # [1] equals in_keys in dict
            clone_dependencies = new_request

            # check if any of our current dependencies already has to fulfil a request
            # since the request's might collide we should just duplicate it
            clone = False
            if clone_instead_merge:
                for d in current_deps:
                    if len(requests.get(d, [])) > 0:
                        clone = True
                        k_in_keys = []

            if (
                new_request[0] is not None
                and "clone_dependencies" in new_request[0]
                and new_request[0]["clone_dependencies"]
            ):
                clone_dependencies = new_request[0]["clone_dependencies"]
                del new_request[0]["clone_dependencies"]
                clone = True
                k_in_keys = []

            if (
                new_request[0] is not None
                and "requires_request" in new_request[0]
                and new_request[0]["requires_request"]
            ):
                del new_request[0]["requires_request"]
                # input_requests[k] = NestedFrozenDict(new_request[0])
                input_requests[k] = new_request[0]

            # all_deps = get_all_dependencies()

            for clone_id, request in enumerate(clone_dependencies):
                if clone:
                    to_clone_keys = dsk_dict[k][1]
                    if not isinstance(to_clone_keys, list):
                        to_clone_keys = [to_clone_keys]

                    # create new node in graph containing k_in_keys as input
                    if insert_predecessor:
                        pre_function = insert_predecessor[clone_id]
                        pre_request = clone_dependencies[clone_id]

                        pre_k = tokenize([k, "foreal_pre", clone_id])
                        pre_base_name = None
                        if hasattr(pre_function, "__self__") and hasattr(
                            pre_function.__self__, "dask_key_name"
                        ):
                            pre_base_name = pre_function.__self__.dask_key_name
                            pre_k = pre_base_name + KEY_SEP + pre_k

                        # # go trough request and update the name of the clone
                        # if pre_base_name is not None:
                        #     update_key_in_config(pre_request,pre_base_name,pre_k)

                        requests[pre_k] = [pre_request]
                        dsk_dict[pre_k] = [pre_function, None, {}]
                        pre_in_keys = []

                    for to_clone_key in to_clone_keys:
                        if insert_predecessor:
                            if to_clone_key is None:
                                pre_in_keys.append(None)
                            else:
                                # pre_in_keys.append(
                                #     base_name(to_clone_key)
                                #     + KEY_SEP
                                #     + tokenize([to_clone_key, "foreal_clone", clone_id])
                                # )
                                pre_in_keys.append(generate_clone_key(k,to_clone_key,clone_id))
                        else:
                            if to_clone_key is None:
                                k_in_keys.append(None)
                            else:
                                # k_in_keys.append(
                                #     base_name(to_clone_key)
                                #     + KEY_SEP
                                #     + tokenize([to_clone_key, "foreal_clone", clone_id])
                                # )
                                k_in_keys.append(generate_clone_key(k,to_clone_key,clone_id))

                    if insert_predecessor:
                        dsk_dict[pre_k][1] = pre_in_keys
                        dsk_dict[pre_k] = tuple(dsk_dict[pre_k])

                        k_in_keys += [pre_k]
                        remove[pre_k] = False
                        new_work[pre_k] = True
                        # new_work.append(pre_k)

                for i, d in enumerate(current_deps):
                    # duplicate keys

                    if clone:
                        clone_work = [d]
                        # d = (
                        #     base_name(d)
                        #     + KEY_SEP
                        #     + tokenize([d, "foreal_clone", clone_id])
                        # )
                        d = generate_clone_key(k,d,clone_id)
                        while clone_work:
                            new_clone_work = []
                            for cd in clone_work:
                                # clone_d = (
                                #     base_name(cd)
                                #     + KEY_SEP
                                #     + tokenize([cd, "foreal_clone", clone_id])
                                # )
                                clone_d = generate_clone_key(k,cd,clone_id)

                                # update_key_in_config(request,cd,clone_d)
                                # TODO: do we need to reset the dask_key_name of each
                                #       of each cloned node?

                                cloned_cd_node = copy(dsk_dict[cd])
                                # cloned_cd_node = deepcopy(dsk_dict[cd])
                                to_clone_keys = cloned_cd_node[1]
                                if not isinstance(to_clone_keys, list):
                                    to_clone_keys = [to_clone_keys]
                                cd_in_keys = []
                                for to_clone_key in to_clone_keys:
                                    if to_clone_key is None:
                                        cd_in_keys.append(None)
                                    else:
                                        # cd_in_keys.append(
                                        #     base_name(to_clone_key)
                                        #     + KEY_SEP
                                        #     + tokenize(
                                        #         [to_clone_key, "foreal_clone", clone_id]
                                        #     )
                                        # )
                                        cd_in_keys.append(generate_clone_key(k,to_clone_key,clone_id))
                                if len(cd_in_keys) == 1:
                                    cd_in_keys = cd_in_keys[0]
                                nd = list(cloned_cd_node)
                                nd[1] = cd_in_keys
                                cloned_cd_node = tuple(nd)
                                dsk_dict[clone_d] = cloned_cd_node
                                new_deps = get_dependencies(dsk_dict, cd, as_list=True)
                                new_clone_work += new_deps
                            clone_work = new_clone_work

                    # determine what needs to be removed
                    if not insert_predecessor:
                        # we are not going to remove anything if we inserted a predecessor node before current node k
                        # we are also not updating the requests of dependencies of the original node k
                        # since it will be done in the next interaction by configuring the inserted predecessor

                        to_be_removed = False
                        if k in remove:
                            to_be_removed = remove[k]
                        if request is None:
                            to_be_removed = True
                        if "remove_dependencies" in request:
                            to_be_removed = request["remove_dependencies"]
                            del request["remove_dependencies"]

                        # TODO: so far this doesn't allow to clone dependencies and delete only one of them
                        #       it might be irrelevant.
                        if request.get("remove_dependency", {}).get(
                            base_name(d), False
                        ):
                            to_be_removed = True
                            del request["remove_dependency"][base_name(d)]

                        if not request.get("remove_dependency", True):
                            # clean up if an empty dict still exists
                            del request["remove_dependency"]
                        if d in requests:
                            if not to_be_removed:
                                requests[d] += [request]
                            remove[d] = remove[d] and to_be_removed
                        else:
                            if not to_be_removed:
                                requests[d] = [request]
                            # if we received None
                            remove[d] = to_be_removed

                        # only configure each node once in a round!
                        # if d not in new_work and d not in work:  # TODO: verify this
                        #     new_work.append(
                        #         d
                        #     )  # TODO: Do we need to configure dependency if we'll remove it?
                        # we should also add `d`` only to the work list if we did not insert a
                        # a predecessor. The predecessor will take care of adding it in the next round
                        # otherwise it could happen that the predecessor changes the name of the node
                        # by cloning it. Then we'd have a deprecated node name in the work list
                        if d not in work and (d not in remove or not remove[d]):
                            new_work[d] = True
            dsk_k = list(dsk_dict[k])
            if k_in_keys is not None:
                if len(k_in_keys) == 1:
                    k_in_keys = k_in_keys[0]
                dsk_k[1] = k_in_keys
            dsk_dict[k] = tuple(dsk_k)

        work = new_work

    # Assembling the configured new graph
    out = {k: dsk_dict[k] for k in out_keys if not remove[k]}

    def clean_request(x):
        """Removes any leftovers of foreal configuration"""
        for item in ["remove_dependencies", "clone_dependencies", "insert_predecessor"]:
            x.pop(item, None)

        return x

    # After we have acquired all requests we can input the required_requests as a input node to the requiring node
    # we assume that the last argument is the request
    for k in input_requests:
        if k not in out:
            continue
        # input_requests[k] = clean_request(input_requests[k])
        # Here we assume that we always receive the same tuple of (bound method, data, request)
        # If the interface changes this will break #TODO: check for all cases
        if isinstance(out[k][2], tuple):
            # FIXME: find a better inversion of unpack_collections().
            #        this is very fragile
            # Check if we've already got a request as argument
            # This is the case if our node will make use of a general config
            # Then the present request is updated with the configured one
            # We need to recreate the tuple/list elements though. (dask changed)
            # TODO: use a distinct request class
            if out[k][2][0] == dict:
                my_dict = {}
                # FIXME: it does not account for nested structures
                for item in out[k][2][1]:
                    if isinstance(item[1], tuple):
                        if item[1][0] == tuple:
                            item[1] = tuple(item[1][1])
                        elif item[1][0] == list:
                            item[1] = list(item[1][1])
                    my_dict[item[0]] = item[1]
                my_dict = {item[0]: item[1] for item in out[k][2][1]}
                my_dict.update(input_requests[k])
                out[k] = out[k][:2] + (my_dict,)
            else:
                # replace the last entry
                out[k] = out[k][:2] + (input_requests[k],)
        elif isinstance(out[k][2], dict):
            out[k][2].update(input_requests[k])
        else:
            # replace the last entry
            out[k] = out[k][:2] + (input_requests[k],)
        # TODO: we might dask.delayed(out[k][2]) here

    # convert to delayed object
    in_keys = list(flatten(keys))

    def dask_optimize(dsk, keys):
        dsk1, deps = cull(dsk, keys)
        dsk2 = inline(dsk1, dependencies=deps)
        # dsk3 = inline_functions(dsk2, keys, [len, str.split],
        #                        dependencies=deps)
        dsk4, deps = fuse(dsk2)
        return dsk4

    #    dsk, dsk_keys = dask.base._extract_graph_and_keys([collection])

    # out = optimize_functions(out, in_keys)
    #    collection = Delayed(key=dsk_keys, dsk=collection)

    if len(in_keys) > 1:
        collection = [Delayed(key=key, dsk=out) for key in in_keys]
    else:
        collection = Delayed(key=in_keys[0], dsk=out)
        if isinstance(delayed, list):
            collection = [collection]

    if optimize_graph:
        collection = optimize(collection, keys)
    #
    return collection


def optimize(delayed, keys=None, dask_optimize=False):
    """Optimizes the graph after configuration"""

    # TODO: Why is this necessary?
    if not isinstance(delayed, list):
        collections = [delayed]
    else:
        collections = delayed

    dsk, dsk_keys = dask.base._extract_graph_and_keys(collections)

    dsk_dict = {k: dsk[k] for k in dsk.keys()}

    if keys is None:
        keys = dsk_keys

    if not isinstance(keys, (list, set)):
        keys = [keys]

    work = list(set(flatten(keys)))

    # Invert the task graph: make a forward graph
    dsk_inv = {k: {} for k in work}

    out_keys = []
    sources = set()
    seen = set()
    while work:
        new_work = []

        out_keys += work
        deps = []
        for k in work:
            current_deps = get_dependencies(dsk_dict, k, as_list=True)

            if not current_deps:
                sources.add(k)

            for i, d in enumerate(current_deps):
                if d not in dsk_inv:
                    dsk_inv[d] = {k}
                else:
                    dsk_inv[d].add(k)

                if d not in seen:
                    new_work.append(d)
                    seen.add(d)

        work = new_work

    def replace(s, r, n):
        if isinstance(s, str) and s == r:
            return n
        return s

    # traverse the task graph in forward direction

    # starting from the sources
    work = list(sources)
    in_keys = list(set(flatten(keys)))
    out_keys = []
    seen = set()
    # import time
    # debug={}
    rename = {}
    while work:
        new_work = []

        deps = []
        for k in work:
            if k in in_keys:
                # if we are at a sink node we don't change the name
                out_keys += [k]
                continue

            # rename k
            node_token = ""
            if hasattr(dsk_dict[k][0], "__self__"):
                node_token = dsk_dict[k][0].__self__

            input_token = list(dsk_dict[k][1:])
            # start = time.time()
            new_k = base_name(k) + KEY_SEP + tokenize([node_token, input_token])
            # duration = time.time() - start
            # debug[base_name(k)] = debug.get(base_name(k),0) + duration

            out_keys += [new_k]
            rename[k] = new_k

            current_dependants = [d for d in dsk_inv[k]]

            for i, d in enumerate(current_dependants):
                # TODO: is there a way to not change the dsk_dict in-place?
                if isinstance(dsk_dict[d][1], list):
                    in1 = [replace(s, k, new_k) for s in dsk_dict[d][1]]
                elif isinstance(dsk_dict[d][1], str):
                    in1 = replace(dsk_dict[d][1], k, new_k)
                else:
                    in1 = dsk_dict[d][1]

                dsk_dict[d] = tuple([dsk_dict[d][0], in1] + list(dsk_dict[d][2:]))

                if d not in seen:
                    new_work.append(d)
                    seen.add(d)

        work = new_work

    for k in rename:
        dsk_dict[rename[k]] = dsk_dict[k]

    out = {k: dsk_dict[k] for k in out_keys}

    def optimize_functions(dsk, keys):
        dsk1, deps = cull(dsk, keys)
        dsk2 = inline(dsk1, dependencies=deps)
        # dsk3 = inline_functions(dsk2, keys, [len, str.split],
        #                        dependencies=deps)
        dsk4, deps = fuse(dsk2, fuse_subgraphs=True)
        return dsk4

    if dask_optimize:
        out = optimize_functions(out, in_keys)

    if len(in_keys) > 1:
        collection = [Delayed(key=key, dsk=out) for key in in_keys]
    else:
        collection = Delayed(key=in_keys[0], dsk=out)
        if isinstance(delayed, list):
            collection = [collection]
    return collection
