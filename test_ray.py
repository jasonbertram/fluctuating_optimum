import ray
import os

# Connect to Ray cluster
ray.init(address=f"{os.environ['HEAD_NODE']}:{os.environ['RAY_PORT']}",_node_ip_address=os.environ['HEAD_NODE'])

# Check that Ray sees two nodes and their status is 'Alive'
print("Nodes in the Ray cluster:")
print(ray.nodes())

# Check that Ray sees 12 CPUs and 2 GPUs over 2 Nodes
print(ray.available_resources())