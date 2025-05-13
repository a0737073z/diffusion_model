import os
import torch
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel # DDPMScheduler is explicitly used
from peft import get_peft_model, LoraConfig # TaskType can be imported if needed for LoraConfig explicitly

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# 自定義資料集
class ImagePromptDataset(Dataset):
    def __init__(self, image_dir, prompt, transform=None):
        self.image_dir = image_dir
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
        self.image_paths = []
        # 在初始化時過濾掉無法讀取的檔案
        for fname in image_files:
            try:
                img_path = os.path.join(image_dir, fname)
                # 嘗試開啟圖片以驗證
                Image.open(img_path).convert("RGB").close() # 轉換並關閉，僅作驗證
                self.image_paths.append(img_path)
            except Exception as e:
                print(f"Warning: Could not load or verify image {os.path.join(image_dir, fname)}. Error: {e}. Skipping this file.")

        self.prompt = prompt
        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            image = self.transform(image)
            return {"pixel_values": image, "prompt": self.prompt}
        except Exception as e:
            # 理論上，初始化時已過濾，但以防萬一
            print(f"Error during __getitem__ for {self.image_paths[idx]}: {e}. Returning a zero tensor.")
            return {"pixel_values": torch.zeros((3, 512, 512)), "prompt": self.prompt}

# 初始化 pipeline
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16) # .to("cuda") 會在之後處理

# 明確設定 scheduler (推薦)
pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")


# 建立 LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=[
        "to_q", "to_k", "to_v", "to_out.0", # 注意：to_out 經常是 Sequential，所以可能是 to_out.0
        "proj_in", "proj_out", # UNetMidBlock2DCrossAttn
        "ff.net.0.proj", "ff.net.2", # FeedForward
        # 檢查 UNet 中 Linear 層的實際名稱可能更精確，例如：
        # pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn1.to_q.weight.shape (檢查層是否存在)
        # 通常，上面這幾個就涵蓋了很多注意力層。如果需要更細緻，需要檢查模型結構。
        # "linear" 太通用，可能不會被 peft 直接匹配，除非它真的是模組的名字。
        # 如果上述 target_modules 效果不佳，可以考慮只用:
        # target_modules=["to_q", "to_k", "to_v", "to_out.0"] # 或 ["to_q", "to_k", "to_v"]
    ],
    lora_dropout=0.1,
    bias="none",
    # task_type=TaskType.FEATURE_EXTRACTION # 通常不需要，peft 會推斷
)

# 應用 LoRA 到 UNet
# 確保 UNet 是 UNet2DConditionModel 的實例
if isinstance(pipe.unet, UNet2DConditionModel):
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    trainable_params = sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in pipe.unet.parameters())
    print(f"Successfully applied LoRA.")
    print(f"Total UNet parameters: {total_params}")
    print(f"Trainable LoRA parameters: {trainable_params} ({trainable_params/total_params*100:.2f}%)")
    if trainable_params == 0:
        print("Warning: No parameters were made trainable by LoRA. Check your target_modules and LoRA config.")
        print("Common target_modules for Stable Diffusion UNet are related to attention layers' q, k, v, out projections.")
        print("Example: ['to_q', 'to_k', 'to_v', 'to_out.0'] or similar based on your UNet structure.")
else:
    print(f"Error: pipe.unet is not an instance of UNet2DConditionModel. It is {type(pipe.unet)}")
    # Handle error or exit


# 準備資料
image_directory = r"C:\Users\user\Desktop\0424\patchcore_13_V2_train\train\0_good" # 請替換為你的路徑
training_prompt = "a clean, high-resolution photo of an industrial machine part, metallic texture, studio lighting, highly detailed, professional catalog photo"

if not os.path.exists(image_directory):
    raise FileNotFoundError(f"Image directory not found: {image_directory}")

train_dataset = ImagePromptDataset(image_directory, prompt=training_prompt)
if len(train_dataset) == 0:
    raise ValueError(f"No valid images found in the directory: {image_directory}. Please check image paths and formats.")

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True) # 建議增加 batch_size

# 訓練參數
# **修正: 只優化可訓練的 LoRA 參數**
lora_trainable_params = [p for p in pipe.unet.parameters() if p.requires_grad]
if not lora_trainable_params:
    raise ValueError("No trainable LoRA parameters found. LoRA might not have been applied correctly.")
optimizer = AdamW(lora_trainable_params, lr=1e-4)


# 訓練迴圈
pipe.unet.train() # 確保 UNet 處於訓練模式
# 如果 text_encoder 也參與了 LoRA (雖然此處配置沒有)，則也應設為 train()
# pipe.text_encoder.train() # 如果 text_encoder 也被 LoRA 修改

print("Starting LoRA training...")
num_epochs = 10 # 可自行調整 epoch 數
for epoch in range(num_epochs):
    total_loss = 0
    for i, batch in enumerate(train_dataloader):
        if batch["pixel_values"].nelement() == 0: # 跳過由 __getitem__ 錯誤處理返回的空 tensor
             print(f"Skipping batch {i} due to empty pixel_values (likely image loading error).")
             continue

        pixel_values = batch["pixel_values"].to("cuda").half()
        prompts_list = batch["prompt"] # prompts_list is a list of strings

        # VAE 編碼
        # 不需要 .to(dtype=torch.float16) 因為 pipe.vae 已經是 float16 (如果 pipeline 以 float16 載入)
        latents = pipe.vae.encode(pixel_values).latent_dist.sample() * pipe.vae.config.scaling_factor # 使用 config 中的 scaling_factor

        # 創建噪聲
        noise = torch.randn_like(latents).to("cuda")
        timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.size(0),), device="cuda").long()

        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

        # 文字編碼
        prompt_input_ids = pipe.tokenizer(
            prompts_list,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to("cuda")
        encoder_hidden_states = pipe.text_encoder(prompt_input_ids)[0] # .to(dtype=torch.float16) if necessary and text_encoder is not already in f16

        # 預測噪聲
        # **修正: .sample 是屬性，不是方法**
        noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # 計算損失
        loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="mean") # reduction="mean" 是預設

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if (i + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_dataloader)} - Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

# 儲存 LoRA 權重
save_directory = "lora_sd_model"
os.makedirs(save_directory, exist_ok=True)

# `save_pretrained` 會儲存適配器權重 (adapter_model.bin) 和配置文件 (adapter_config.json)
# 如果 UNet 是 PeftModel，它有 save_pretrained 方法
if hasattr(pipe.unet, 'save_pretrained'):
    pipe.unet.save_pretrained(save_directory)
    # 如果 text_encoder 也被 LoRA 修改並想保存
    # if hasattr(pipe.text_encoder, 'save_pretrained'):
    # pipe.text_encoder.save_pretrained(os.path.join(save_directory, "text_encoder_lora"))
    print(f"LoRA fine-tuning complete! Weights saved to {save_directory}")
else:
    print("Error: pipe.unet does not have save_pretrained. It might not be a PeftModel.")


# 推論時如何載入 (範例):
# from diffusers import StableDiffusionPipeline
# from peft import PeftModel
#
# model_id = "runwayml/stable-diffusion-v1-5"
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe.unet = PeftModel.from_pretrained(pipe.unet, save_directory) # 載入 LoRA 權重到 UNet
# pipe = pipe.to("cuda")
#
# prompt = "a photo of <your-concept> in the style of <something>" # 替換成你的提示
# image = pipe(prompt).images[0]
# image.save("lora_generated_image.png")