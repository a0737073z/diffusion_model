from diffusers import StableDiffusionUpscalePipeline
from PIL import Image
import torch
import requests
from huggingface_hub import configure_http_backend

#跳過SSL驗證
def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)
# 載入 Upscaler 管線
pipe = StableDiffusionUpscalePipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# 載入原始圖片（例如 128x128）
low_res_img = Image.open(r"C:\Users\user\Desktop\0422相機校正\NG_scratch_054.png").convert("RGB")
low_res_img = low_res_img.resize((512, 512))  # 注意：輸入建議在 128x128～512x512 之間

# 提供一個提示詞（可以用簡單的，比如 “a photo”）
prompt = "a photo"

# 執行放大
upscaled_image = pipe(prompt=prompt, image=low_res_img).images[0]

# 儲存與顯示
upscaled_image.save(r"C:\Users\user\Desktop\result\upscaled_output_2.png")
upscaled_image.show()