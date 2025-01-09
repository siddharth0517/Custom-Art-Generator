import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

# Load the Stable Diffusion model
@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    # pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

pipe = load_model()

# Streamlit UI
st.title("Custom Art Generator 🎨")
st.subheader("Generate stunning artwork from your imagination!")

# Input prompt
prompt = st.text_input("Enter your art description (e.g., 'A fantasy landscape with a castle and dragon')")

# Additional settings
guidance_scale = st.slider("Creativity Level (Higher values stick more to the prompt)", 7.0, 15.0, 7.5, step=0.5)
num_steps = st.slider("Image Quality (More steps take longer)", 20, 100, 50, step=10)

# Generate button
if st.button("Generate Artwork"):
    if prompt:
        with st.spinner("Creating your masterpiece..."):
            image = pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=num_steps).images[0]
            st.image(image, caption="Generated Artwork", use_column_width=True)
            
            # Save the generated image
            image.save("generated_artwork.png")
            st.success("Artwork saved as 'generated_artwork.png'!")
    else:
        st.warning("Please enter a description to generate artwork.")
