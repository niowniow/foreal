from dash import html


class ForealControl:
    def __init__(self, layout=None, elements_to_requests_mapping=None):
        """The control class enables a developer to add custom controls to the foreal
           webapp. It is designed to transform the values of GUI elements
           into a foreal request.

        Args:
            layout (dash layout, optional): contains any custom dash elements (like buttons,inputs,etc). Defaults to None.
            elements_to_requests_mapping (dict, optional): The dicts' keys are ids of dash elements which should trigger a call of the control function. The value of the dicts' elements must be a tuple ('value','request_key'). `value` contains the property of the dash element that should be mapped to the `request_key` of the request. When None it defaults to an empty dict.
        """
        if layout is None:
            layout = html.Div([])
        self.layout = layout

        if elements_to_requests_mapping is None:
            elements_to_requests_mapping = {}
        self.elements_to_requests_mapping = elements_to_requests_mapping

    def control(self, request):
        for id in self.elements_to_requests_mapping:
            if id in request["self"]:
                value = request["self"][id]
                # get the request_key
                dict_key = self.elements_to_requests_mapping[id][1]

                if not value:
                    if dict_key in request:
                        del request[dict_key]
                else:
                    request[dict_key] = value

        return request
