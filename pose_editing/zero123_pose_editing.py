import os
import subprocess

def edit_pose(image, azimuth, polar):
    # Save the current working directory
    original_cwd = os.getcwd()
    
    # Change to the 'threestudio' directory
    os.chdir('threestudio')
    
    # Build the command
    command = [
        "python",
        "launch.py",
        "--config",
        "custom/threestudio-mvimg-gen/configs/stable-zero123.yaml",
        "--test",
        "--gpu",
        "0",
        f"data.image_path={image}",
        f"data.default_elevation_deg={polar}",
        f"data.default_azimuth_deg={azimuth}"
    ]
    
    # Run the command
    subprocess.run(command)
    
    # Change back to the original directory
    os.chdir(original_cwd)

# Call the function
# edit_pose('../segmented_rgba.png', 72.0, -5.0)
