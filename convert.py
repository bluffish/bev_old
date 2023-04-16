import torch

model_state_dict = torch.load("pretrained/model525000.pt")

updated_state_dict = {}
for key, value in model_state_dict.items():
    # Split the key using '.' as the delimiter
    key_parts = key.split('.')

    # Replace "backbone" with "module" only if it's the first layer
    if key_parts[0] == "backbone":
        key_parts[0] = "module"
        new_key = '.'.join(key_parts)
    else:
        key_parts.insert(0, "module")
        new_key = '.'.join(key_parts)
        # new_key = key

    updated_state_dict[new_key] = value
torch.save(updated_state_dict, "pretrained/model_converted_headless.pt")
print(updated_state_dict)