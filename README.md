Root Folder: Avatar Assignment

\#Create conda environment:  
conda create \-n avatar python=3.10  
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 \-c pytorch \-c nvidia

\#Install python dependencies using: 

pip install \-r requirements.txt

\#Install threestudio for stablezero123 [GitHub \- threestudio-project/threestudio: A unified framework for 3D content generation.](https://github.com/threestudio-project/threestudio?tab=readme-ov-file) [GitHub \- DSaurus/threestudio-mvimg-gen](https://github.com/DSaurus/threestudio-mvimg-gen)  
Download the stable zero123 checkpoint from https://huggingface.co/stabilityai/stable-zero123/blob/main/stable\_zero123.ckpt and store it in threestudio/load/zero123

cd threestudio

pip install \-r requirements.txt

For tinycudann to install without issues:  
Set export CUDA\_HOME=/usr/lib/cuda

config.yaml:  
Edit the following in config

root\_dir: "results/sofa\_a50\_p15\_v2"  \# Set the root directory for results

segmentation:  
  method: "sam"    
  model: "facebook/sam-vit-base"

pose\_editing:  
  method: "zero123"  

inpainting:  
  method: "stable\_diffusion"  
   
input\_image: "./sofa.jpg"  
inpaint\_output\_image: "${root\_dir}/inpaint\_output\_image.png"  
mask\_image: "${root\_dir}/segmentation\_mask.png"  
segmented\_image: "${root\_dir}/segmented\_rgba.png"  
pose\_rotated\_mask: "${root\_dir}/pose\_rotated\_object\_mask.png"  
pose\_edited\_image: "${root\_dir}/pose\_edited\_image.png"  
inpaint\_mask: "${root\_dir}/inpaint\_mask.png"

\# User input for LLM-based entity extraction  
user\_input: "Rotate the sofa chair by azimuth 50 degrees and polar 15 degrees."

\# Optional quantization flag for LLM  
quantize: False  \# Set to True to enable 8-bit quantization

prompt: "Strictly restore the missing parts of white wall, floor, borders of a single chair sofa facing sideways to complete the scene. high resolution, smooth sharp and consistent"  
negative\_prompt: "artifacts,chair,sofa,furniture"  
guidance\_scale: 8.0

Run python main.py to get the results. inpaint\_output\_image.png is the final output.

**Work Flow:**

1. **Entity Extraction: Use Phi3.5 to extract the object name, azimuth angle and polar angle from the text**  
2. **Segmentation: Use Grounding Dino to get the bounding box for the object using text prompt, further use this as grounding for segment anything model to get the mask.**  
3.  **StableZero123: use stable zero123 to get the 3d object from image and novel view synthesis.**  
4. **Inpainting: Use stable diffusion 2 inpainting model with dilation, prompt, negative prompt and guidance scale. Tune these parameters for better results.** 

Check Report.pdf for results
