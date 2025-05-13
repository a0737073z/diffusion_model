from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import requests
from huggingface_hub import configure_http_backend

#解決SSL憑證問題
def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)

# 載入模型
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

# 載入圖片（你也可以用 Image.open("path/to/image.png")）
init_image = Image.open(r"C:\Users\user\Desktop\0424\patchcore_13_V2_train\train\0_good\3R_1.png").convert("RGB").resize((1392, 1008))

# 設定提示詞與參數
prompt = "a clean, high-resolution photo of an industrial machine part, metallic texture, studio lighting, highly detailed, professional catalog photo"
generator = torch.Generator("cuda").manual_seed(42)

# 開始生成（strength 越高越偏離原圖）
image = pipe(
    prompt=prompt,
    image=init_image,
    strength=0.4,
    guidance_scale=7.5,
    generator=generator,
    num_inference_steps=100,
).images[0]

# 儲存圖片
image.save(r"C:\Users\user\Desktop\0424\PatchCore_diffusion_data\img2img_output_4.png")
