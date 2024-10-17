import yaml
import os

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Extract the root_dir value
    root_dir = config.get('root_dir', '.')

    # Replace variables with actual root_dir path
    for key, value in config.items():
        if isinstance(value, str) and "${root_dir}" in value:
            config[key] = value.replace("${root_dir}", root_dir)
    
    os.makedirs(root_dir, exist_ok=True)

    print(config['mask_image'])
    return config
