from tools.tf2ngraph import Tf2ngraphJson


json_name = 'test.json'

# string to string dictionary
optional_params = {"max_cores":"12", "abc":"2"}

# list of dictionaries. each dictionary has string keys (node names) and values are list of integers
shape_hints = [{"a":[1,-1], "b:":[1,-1]}, {"a":[2,-1], "b:":[2,-1]}]
Tf2ngraphJson.dump_json(json_name, optional_params=optional_params, shape_hints=shape_hints)