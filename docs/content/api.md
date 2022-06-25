# API Documentation

## Configuration
The function relevant for configuring the task graph and running it are `foreal.compute`. It calls the `configuration` function which itself progagates through the graph and calls each Node's `configure` function. 

@pydoc foreal.convenience.compute

@pydoc foreal.core.graph.configuration

@pydoc foreal.core.graph.Node
