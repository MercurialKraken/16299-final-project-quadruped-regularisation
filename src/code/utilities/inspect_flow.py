import torch
c = torch.load('A:/IsaacLab/flow_model_adaptive.pt', map_location='cpu', weights_only=True)
print("type:", type(c))
if hasattr(c, 'keys'):
    keys = list(c.keys())
    print("keys:", keys[:10])
    if 'state_dim' in c:
        print("state_dim:", c['state_dim'])
    if 'state_dict' in c:
        print("state_dict first key:", list(c['state_dict'].keys())[0])
        print("first weight shape:", next(iter(c['state_dict'].values())).shape)
    else:
        print("first weight shape:", next(iter(c.values())).shape)
