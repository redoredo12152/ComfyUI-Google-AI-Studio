"""
Google AI Studio Nodes for ComfyUI
https://github.com/BuffMcBigHuge/ComfyUI-Google-AI-Studio
"""

import io
import os
import tempfile
import wave
import base64
import folder_paths
import numpy as np
import torch
from typing import Dict, Any, Tuple, List

try:
    from google import genai
    from google.genai import types
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    print("Google AI SDK not available. Please install with: pip install google-genai")


class GeminiPromptSplitter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                # 기본 구분자를 [NEGATIVE]로 설정했습니다.
                "separator": ("STRING", {"default": "[NEGATIVE]"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "split_prompt"
    CATEGORY = "Google-AI-Studio"

    def split_prompt(self, text, separator):
        # [NEGATIVE] 태그가 문장 안에 있으면 그 기준으로 쪼갭니다.
        if separator in text:
            parts = text.split(separator, 1)
            positive = parts[0].strip()
            negative = parts[1].strip()
        else:
            # 태그가 없으면 전부 긍정 프롬프트로 몰아넣습니다.
            positive = text.strip()
            negative = ""
            
        return (positive, negative)


class GoogleAIStudioTTSNode:
    """
    Google AI Studio Text-to-Speech Node
    Converts text to speech using Google's Gemini TTS models
    """
    
    # Available voices from the 2025 documentation with styles
    VOICES = [
        "Zephyr (Bright)", "Puck (Upbeat)", "Charon (Informative)", 
        "Kore (Firm)", "Fenrir (Excitable)", "Leda (Youthful)",
        "Orus (Firm)", "Aoede (Breezy)", "Callirrhoe (Easy-going)", 
        "Autonoe (Bright)", "Enceladus (Breathy)", "Iapetus (Clear)",
        "Umbriel (Easy-going)", "Algieba (Smooth)", "Despina (Smooth)", 
        "Erinome (Clear)", "Algenib (Gravelly)", "Rasalgethi (Informative)",
        "Laomedeia (Upbeat)", "Achernar (Soft)", "Alnilam (Firm)", 
        "Schedar (Even)", "Gacrux (Mature)", "Pulcherrima (Forward)",
        "Achird (Friendly)", "Zubenelgenubi (Casual)", "Vindemiatrix (Gentle)", 
        "Sadachbia (Lively)", "Sadaltager (Knowledgeable)", "Sulafat (Warm)"
    ]
    
    MODELS = [
        "gemini-2.5-flash-preview-tts",
        "gemini-2.5-pro-preview-tts"
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello! This is a test of Google AI Studio text-to-speech."
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Your Google AI Studio API key"
                }),
                "model": (cls.MODELS, {
                    "default": "gemini-2.5-flash-preview-tts"
                }),
                "voice": (cls.VOICES, {
                    "default": "Kore (Firm)"
                }),
            },
            "optional": {
                "instruction_prefix": ("STRING", {
                    "default": "Say cheerfully:",
                    "tooltip": "Optional instruction prefix for style control"
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "Google AI Studio"
    DESCRIPTION = "Convert text to speech using Google AI Studio TTS models"

    def generate_speech(self, text: str, api_key: str, model: str, voice: str, instruction_prefix: str = "") -> Tuple[Dict[str, Any]]:
        """
        Generate speech from text using Google AI Studio TTS
        """
        if not GOOGLE_AI_AVAILABLE:
            raise Exception("Google AI SDK not installed. Please install with: pip install google-genai")
        
        if not api_key.strip():
            raise Exception("API key is required. Get one from https://aistudio.google.com/")
        
        try:
            # Set up the API key
            os.environ['GOOGLE_API_KEY'] = api_key.strip()
            
            # Initialize the client
            client = genai.Client()
            
            # Prepare the content with optional instruction prefix
            content = f"{instruction_prefix} {text}".strip() if instruction_prefix.strip() else text
            
            # Extract voice name from "Voice (Style)" format
            voice_name = voice.split(' (')[0] if ' (' in voice else voice
            
            # Generate the speech
            response = client.models.generate_content(
                model=model,
                contents=content,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice_name,
                            )
                        )
                    ),
                )
            )
            
            # Extract audio data
            audio_data = response.candidates[0].content.parts[0].inline_data.data
            
            # Convert to audio format compatible with ComfyUI
            audio_dict = self._convert_audio_data(audio_data)
            
            return (audio_dict,)
            
        except Exception as e:
            raise Exception(f"TTS generation failed: {str(e)}")
    
    def _convert_audio_data(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Convert raw audio data to ComfyUI audio format
        """
        temp_path = None
        try:
            # Create a temporary file to process the audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Write the audio data as WAV
            self._write_wave_file(temp_path, audio_data)
            
            # Read back the audio data for ComfyUI
            with wave.open(temp_path, 'rb') as wav_file:
                nframes = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frames = wav_file.readframes(nframes)
                
                # Determine numpy dtype based on sample width
                if sample_width == 1:
                    dtype = np.uint8
                elif sample_width == 2:
                    dtype = np.int16
                elif sample_width == 4:
                    dtype = np.int32
                else:
                    dtype = np.int16
                
                # Convert to numpy array
                audio_array = np.frombuffer(frames, dtype=dtype)
                
                # Normalize to [-1, 1] range for ComfyUI
                if dtype == np.uint8:
                    audio_array = (audio_array.astype(np.float32) - 128) / 128.0
                elif dtype == np.int16:
                    audio_array = audio_array.astype(np.float32) / 32768.0
                elif dtype == np.int32:
                    audio_array = audio_array.astype(np.float32) / 2147483648.0
                
                # Reshape for ComfyUI batch format: [batch, channels, samples]
                if channels == 1:
                    # Mono: reshape to [1, 1, samples]
                    audio_array = audio_array.reshape(1, 1, -1)
                elif channels == 2:
                    # Stereo: reshape to [1, 2, samples]
                    audio_array = audio_array.reshape(-1, 2).T
                    audio_array = audio_array.reshape(1, 2, -1)
                else:
                    # Multi-channel: reshape to [1, channels, samples]
                    audio_array = audio_array.reshape(-1, channels).T
                    audio_array = audio_array.reshape(1, channels, -1)
                
                # Convert to PyTorch tensor
                tensor = torch.from_numpy(audio_array).float()
            
            return {
                "waveform": tensor,
                "sample_rate": sample_rate
            }
            
        finally:
            # Clean up temp file with retry logic for Windows
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except (PermissionError, OSError):
                    # On Windows, sometimes files are locked. Try a few times.
                    import time
                    for i in range(3):
                        time.sleep(0.1)
                        try:
                            os.unlink(temp_path)
                            break
                        except (PermissionError, OSError):
                            if i == 2:  # Last attempt, silently fail
                                pass
    
    def _write_wave_file(self, filename: str, pcm_data: bytes, channels: int = 1, rate: int = 24000, sample_width: int = 2):
        """
        Write PCM data to a WAV file
        """
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(pcm_data)


class GoogleAIStudioTextGenNode:
    """
    Google AI Studio Text Generation Node (Advanced Memory Edition)
    Generates text using Google's Gemini models with historical diversity tracking.
    """
    TEXT_MODELS = [
        "gemini-3.5-flash", "gemini-3.1-pro", "gemini-3.1-flash-lite",
        "gemini-3-flash", "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
    ]
    
    # 👑 [메모리 저장소] 최근 성공한 프롬프트들을 순서대로 담는 리스트(Queue)
    history_buffer = []

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "prompt": ("STRING", {"multiline": True, "default": "Write a creative short story about artificial intelligence."}),
                    "api_key": ("STRING", {"default": "", "tooltip": "Your Google AI Studio API key"}),
                    "model": (cls.TEXT_MODELS, {"default": "gemini-3.1-flash-lite"}),
                },
                "optional": {
                    "system_instruction": ("STRING", {"multiline": True, "default": "", "tooltip": "System instruction to guide the model's behavior"}),
                    "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1, "tooltip": "Controls randomness in generation (0=deterministic, 2=very creative)"}),
                    "max_output_tokens": ("INT", {"default": 8192, "min": 1, "max": 8192, "step": 1, "tooltip": "Maximum number of tokens to generate"}),
                    "thinking_level": (["off", "low", "medium", "high"], {"default": "off", "tooltip": "Reasoning depth (Gemini 2.5/3 only)."}),
                    "safety_filter": ([types.HarmBlockThreshold.OFF, types.HarmBlockThreshold.BLOCK_NONE, types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE, types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE, types.HarmBlockThreshold.BLOCK_ONLY_HIGH], {"default": types.HarmBlockThreshold.BLOCK_NONE}),
                    
                    # 🎲 다양성 강제 모드
                    "diversity_mode": (["off", "on"], {"default": "off", "tooltip": "If on, forces the model to generate different options from previous outputs."}),
                    
                    # 📚 기억할 이전 응답 개수 (기본 5개, 최대 20개까지 설정 가능)
                    "history_count": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1, "tooltip": "Number of previous outputs to remember for diversity generation."}),
                }
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_text"
    CATEGORY = "Google AI Studio"
    DESCRIPTION = "Generate text using Google AI Studio models with History Memory"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 무한 랜덤 생성을 위해 매번 새로운 값(NaN) 반환
        return float("NaN")

    def generate_text(self, prompt: str, api_key: str, model: str, system_instruction: str = "", temperature: float = 0.7, max_output_tokens: int = 1024, thinking_level: str = "off", safety_filter: str = "OFF", diversity_mode: str = "off", history_count: int = 5) -> tuple:
        import time 
        
        if not GOOGLE_AI_AVAILABLE:
            raise Exception("Google AI SDK not installed. Please install with: pip install google-genai")
        if not api_key.strip():
            raise Exception("API key is required. Get one from https://aistudio.google.com/")

        # ==========================================
        # 🎲 다중 기억 기반 다양성 강제 (Diversity Enforcement) 로직
        # ==========================================
        final_prompt = prompt
        
        # diversity_mode가 on이고, 메모리에 1개 이상의 이전 응답이 있을 때 발동
        if diversity_mode == "on" and len(GoogleAIStudioTextGenNode.history_buffer) > 0:
            
            # 저장된 이전 응답들을 보기 좋게 [1], [2] 번호를 매겨 하나의 텍스트로 묶음
            history_text = "\n\n".join([f"[{i+1}] {text}" for i, text in enumerate(GoogleAIStudioTextGenNode.history_buffer)])
            
            diversity_msg = (
                "\n\n[CRITICAL RULE: DIVERSITY ENFORCEMENT]\n"
                "Here is a history of your RECENT outputs. You MUST NOT reuse the exact same specific attributes "
                "(e.g., body type, clothing, pose, background, ethnicity) that were used in ANY of these past generations. "
                "You must strictly force yourself to choose COMPLETELY DIFFERENT options for this new generation.\n\n"
                f"--- PAST OUTPUT HISTORY ---\n{history_text}\n---------------------------\n\n"
                "Now, generate a newly randomized prompt following the original rules."
            )
            final_prompt = final_prompt + diversity_msg

        # ==========================================
        # ⚙️ 에러 방어 및 재시도(Retry) 설정
        # ==========================================
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                os.environ['GOOGLE_API_KEY'] = api_key.strip()
                client = genai.Client()

                config_kwargs = dict(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    safety_settings=[
                        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=safety_filter),
                        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=safety_filter),
                        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=safety_filter),
                        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=safety_filter)
                    ]
                )
                
                if thinking_level != "off":
                    config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level.upper())
                    
                generation_config = types.GenerateContentConfig(**config_kwargs)

                if system_instruction.strip():
                    generation_config.system_instruction = system_instruction.strip()

                # 📡 Gemini API 호출
                response = client.models.generate_content(model=model, contents=final_prompt, config=generation_config)
                generated_text = response.text if response.text else ""
                
                # 🎯 성공 처리: 큐(Queue) 방식으로 버퍼 밀어내기
                if generated_text.strip():
                    GoogleAIStudioTextGenNode.history_buffer.append(generated_text.strip())
                    
                    # 설정한 개수(history_count)를 초과하면 가장 오래된(첫 번째) 응답을 버림
                    if len(GoogleAIStudioTextGenNode.history_buffer) > history_count:
                        GoogleAIStudioTextGenNode.history_buffer.pop(0)
                    
                return (generated_text,)

            except Exception as e:
                print(f"⚠️ [API 시도 {attempt + 1}/{max_retries}] 에러 발생: {e}")
                
                if attempt < max_retries - 1:
                    print(f"🔄 {retry_delay}초 후 다시 시도합니다...")
                    time.sleep(retry_delay)
                else:
                    print(f"❌ 모든 재시도가 실패했습니다. 최신 성공 텍스트를 반환하여 에러를 우회합니다.")
                    
                    # 메모리에 저장된 기록이 있다면 가장 마지막(-1) 성공 응답을 반환, 
                    # 쌩초반부터 터졌다면 하드코딩된 에러용 프롬프트를 반환
                    fallback = GoogleAIStudioTextGenNode.history_buffer[-1] if len(GoogleAIStudioTextGenNode.history_buffer) > 0 else "(error_fallback:1.2), highly detailed photo of a woman"
                    return (fallback,)

class GoogleAIStudioImageGenNode:
    """
    Google AI Studio Image Generation Node
    Generates images using Google's Gemini and Imagen models
    """
    
    # Updated per https://ai.google.dev/gemini-api/docs/changelog and deprecations
    # Deprecated: gemini-2.5-flash-image-preview (Feb 2026)
    IMAGE_MODELS = [
        "gemini-3.1-flash-image-preview",            # Gemini 3.1 Flash Image (Nano Banana 2)
        "gemini-3-pro-image-preview",                # Gemini 3 Pro Image (Nano Banana Pro)
        "gemini-2.5-flash-image",                    # Gemini 2.5 Flash Image (Nano Banana)
        "imagen-4.0-fast-generate-001",              # Imagen 4 Fast
        "imagen-4.0-generate-001",                   # Imagen 4 (paid tier)
        "imagen-4.0-ultra-generate-001",             # Imagen 4 Ultra
        "imagen-3.0-fast-generate-001",              # Imagen 3 Fast
    ]
    
    ASPECT_RATIOS = [
        "1:1", "9:16", "16:9", "3:4", "4:3"
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful landscape with mountains and a lake at sunset"
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Your Google AI Studio API key"
                }),
                "model": (cls.IMAGE_MODELS, {
                    "default": "gemini-3.1-flash-image-preview"
                }),
            },
            "optional": {
                "input_image": ("IMAGE", {
                    "tooltip": "Input image for editing/modification (Gemini models only)"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "What you don't want in the image (works with all models)"
                }),
                "aspect_ratio": (cls.ASPECT_RATIOS, {
                    "default": "1:1",
                    "tooltip": "⚠️ IMAGEN ONLY - Completely ignored for Gemini models"
                }),
                "safety_filter_level": (["BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE"], {
                    "default": "BLOCK_MEDIUM_AND_ABOVE",
                    "tooltip": "⚠️ IMAGEN ONLY - Completely ignored for Gemini models (Gemini has its own safety system)"
                }),
                "person_generation": (["DONT_ALLOW", "ALLOW_ADULT", "ALLOW_ALL"], {
                    "default": "ALLOW_ALL",
                    "tooltip": "⚠️ IMAGEN ONLY - Completely ignored for Gemini models"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "Google AI Studio"
    DESCRIPTION = "Generate images using Gemini (free tier, basic controls) or Imagen (paid tier, advanced controls)"

    def _convert_comfyui_image_to_base64(self, image_tensor: torch.Tensor) -> List[str]:
        """
        Convert ComfyUI image tensor(s) to base64 encoded PNG(s)
        Returns a list of base64 strings, one per image in the batch
        """
        from PIL import Image
        import io
        
        # ComfyUI format: [batch, height, width, channels] with values in [0, 1]
        if len(image_tensor.shape) == 3:
            # Single image without batch dimension, add it
            image_tensor = image_tensor.unsqueeze(0)
        
        batch_size = image_tensor.shape[0]
        base64_images = []
        
        for i in range(batch_size):
            # Get single image from batch
            single_image = image_tensor[i]
            
            # Convert to uint8
            image_array = (single_image.cpu().numpy() * 255).astype(np.uint8)
            
            # Ensure RGB format
            if len(image_array.shape) == 2:  # Grayscale
                image_array = np.stack([image_array] * 3, axis=-1)
            elif image_array.shape[2] == 4:  # RGBA
                image_array = image_array[:, :, :3]  # Remove alpha channel
            
            # Create PIL Image
            pil_image = Image.fromarray(image_array, 'RGB')
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()
            
            base64_images.append(base64.b64encode(image_bytes).decode('utf-8'))
        
        return base64_images

    def generate_image(self, prompt: str, api_key: str, model: str,
                      input_image=None, negative_prompt: str = "", aspect_ratio: str = "1:1",
                      safety_filter_level: str = "BLOCK_MEDIUM_AND_ABOVE",
                      person_generation: str = "ALLOW_ALL") -> tuple:
        """
        Generate images using Google AI Studio Gemini or Imagen models
        """
        if not GOOGLE_AI_AVAILABLE:
            raise Exception("Google AI SDK not installed. Please install with: pip install google-genai")
        
        if not api_key.strip():
            raise Exception("API key is required. Get one from https://aistudio.google.com/")
        
        try:
            # Set up the API key
            os.environ['GOOGLE_API_KEY'] = api_key.strip()
            
            # Initialize the client
            client = genai.Client()
            
            # Handle different model types
            if model.startswith("gemini"):
                # Use Gemini's generate_content with image response modality
                # Note: Gemini models don't support Imagen-specific parameters like
                # safety_filter_level, person_generation, or aspect_ratio
                
                # Prepare content parts
                content_parts = []
                
                # Add input images if provided (for image editing/modification)
                if input_image is not None:
                    image_base64_list = self._convert_comfyui_image_to_base64(input_image)
                    # Add all images from the batch to content parts
                    for image_base64 in image_base64_list:
                        content_parts.append(
                            types.Part.from_bytes(
                                data=base64.b64decode(image_base64),
                                mime_type="image/png"
                            )
                        )
                
                # Add text prompt
                full_prompt = prompt
                if negative_prompt.strip():
                    full_prompt += f"\n\nAvoid: {negative_prompt.strip()}"
                content_parts.append(types.Part.from_text(text=full_prompt))
                
                # Create content with multimodal parts
                contents = [
                    types.Content(
                        role="user",
                        parts=content_parts
                    )
                ]
                
                # Gemini models use their own safety system, not Imagen's safety filter levels
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_modalities=['TEXT', 'IMAGE']
                        # Note: No safety_filter_level, person_generation, or aspect_ratio
                        # as these are Imagen-specific parameters
                    )
                )
                
                # Check if response has candidates
                if not response.candidates:
                    # Check for safety ratings or other blocking reasons
                    error_msg = "No candidates in response"
                    if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                        if hasattr(response.prompt_feedback, 'block_reason'):
                            error_msg = f"Request blocked: {response.prompt_feedback.block_reason}"
                        elif hasattr(response.prompt_feedback, 'safety_ratings'):
                            blocked_categories = []
                            for rating in response.prompt_feedback.safety_ratings:
                                if hasattr(rating, 'blocked') and rating.blocked:
                                    blocked_categories.append(str(rating.category))
                            if blocked_categories:
                                error_msg = f"Content blocked by safety filters: {', '.join(blocked_categories)}"
                    raise Exception(f"Gemini API error: {error_msg}")
                
                candidate = response.candidates[0]
                if not candidate.content or not candidate.content.parts:
                    # Check for finish reason
                    finish_reason = "unknown"
                    if hasattr(candidate, 'finish_reason'):
                        finish_reason = candidate.finish_reason
                    raise Exception(f"No content in response. Finish reason: {finish_reason}")
                
                # Extract image from response
                for part in candidate.content.parts:
                    if part.inline_data is not None:
                        from PIL import Image
                        import io
                        
                        # Convert data to image
                        image_data = part.inline_data.data
                        pil_image = Image.open(io.BytesIO(image_data))
                        
                        # Convert PIL to numpy array
                        image_array = np.array(pil_image)
                        
                        # Ensure RGB format
                        if len(image_array.shape) == 2:  # Grayscale
                            image_array = np.stack([image_array] * 3, axis=-1)
                        elif image_array.shape[2] == 4:  # RGBA
                            image_array = image_array[:, :, :3]  # Remove alpha channel
                        
                        # Normalize to [0, 1] range
                        if image_array.dtype == np.uint8:
                            image_array = image_array.astype(np.float32) / 255.0
                        
                        # Add batch dimension for ComfyUI: [batch, height, width, channels]
                        image_tensor = torch.from_numpy(image_array).unsqueeze(0)
                        
                        return (image_tensor,)
                
                raise Exception("No image generated by Gemini model")
                
            else:
                # Use Imagen's generate_images method (note: plural)
                # Generate the image
                response = client.models.generate_images(
                    model=model,
                    prompt=prompt,
                    config=types.GenerateImagesConfig(
                        aspect_ratio=aspect_ratio,
                        safety_filter_level=safety_filter_level,
                        person_generation=person_generation,
                        negative_prompt=negative_prompt.strip() if negative_prompt.strip() else None,
                        number_of_images=1
                    )
                )
                
                # Convert the generated image to ComfyUI format
                if response.generated_images and len(response.generated_images) > 0:
                    image_data = response.generated_images[0]
                    
                    # Convert to PIL Image first
                    from PIL import Image
                    import io
                    
                    # Get image bytes from the response
                    if hasattr(image_data, 'image') and hasattr(image_data.image, 'image_bytes'):
                        image_bytes = image_data.image.image_bytes
                    elif hasattr(image_data, 'image_bytes'):
                        image_bytes = image_data.image_bytes
                    elif hasattr(image_data, 'data'):
                        image_bytes = image_data.data
                    else:
                        raise Exception("No image data found in Imagen response")
                    
                    # Convert bytes to PIL Image
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    
                    # Convert PIL to numpy array
                    image_array = np.array(pil_image)
                    
                    # Ensure RGB format
                    if len(image_array.shape) == 2:  # Grayscale
                        image_array = np.stack([image_array] * 3, axis=-1)
                    elif image_array.shape[2] == 4:  # RGBA
                        image_array = image_array[:, :, :3]  # Remove alpha channel
                    
                    # Normalize to [0, 1] range
                    if image_array.dtype == np.uint8:
                        image_array = image_array.astype(np.float32) / 255.0
                    
                    # Add batch dimension for ComfyUI: [batch, height, width, channels]
                    image_tensor = torch.from_numpy(image_array).unsqueeze(0)
                    
                    return (image_tensor,)
                else:
                    raise Exception("No images were generated by Imagen model")
            
        except Exception as e:
            # Try to extract more detailed error information
            error_details = str(e)
            
            # Check if it's a Google API error with more details
            if hasattr(e, 'details'):
                error_details = f"{error_details} - Details: {e.details}"
            elif hasattr(e, 'message'):
                error_details = f"{error_details} - Message: {e.message}"
            
            # Check for status code if available
            if hasattr(e, 'code'):
                error_details = f"{error_details} - Code: {e.code}"
            
            raise Exception(f"Image generation failed: {error_details}")


class GoogleAIStudioMultiSpeakerTTSNode:
    """
    Google AI Studio Multi-Speaker Text-to-Speech Node
    Converts text with multiple speakers to speech using Google's Gemini TTS models
    """
    
    VOICES = GoogleAIStudioTTSNode.VOICES
    MODELS = GoogleAIStudioTTSNode.MODELS

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transcript": ("STRING", {
                    "multiline": True,
                    "default": "Dr. Anya: Welcome to our show!\nLiam: Thanks for having me, it's great to be here!",
                    "tooltip": "Multi-speaker transcript with speaker names followed by colons"
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Your Google AI Studio API key"
                }),
                "model": (cls.MODELS, {
                    "default": "gemini-2.5-flash-preview-tts"
                }),
                "speaker1_name": ("STRING", {
                    "default": "Dr. Anya",
                    "tooltip": "Name of the first speaker"
                }),
                "speaker1_voice": (cls.VOICES, {
                    "default": "Kore (Firm)"
                }),
                "speaker2_name": ("STRING", {
                    "default": "Liam",
                    "tooltip": "Name of the second speaker"
                }),
                "speaker2_voice": (cls.VOICES, {
                    "default": "Puck (Upbeat)"
                }),
            },
            "optional": {
                "speaker3_name": ("STRING", {
                    "default": "",
                    "tooltip": "Name of the third speaker (optional)"
                }),
                "speaker3_voice": (cls.VOICES, {
                    "default": "Zephyr (Bright)"
                }),
                "speaker4_name": ("STRING", {
                    "default": "",
                    "tooltip": "Name of the fourth speaker (optional)"
                }),
                "speaker4_voice": (cls.VOICES, {
                    "default": "Charon (Informative)"
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_multispeaker_speech"
    CATEGORY = "Google AI Studio"
    DESCRIPTION = "Convert multi-speaker text to speech using Google AI Studio TTS models"

    def generate_multispeaker_speech(self, transcript: str, api_key: str, model: str, 
                                   speaker1_name: str, speaker1_voice: str,
                                   speaker2_name: str, speaker2_voice: str,
                                   speaker3_name: str = "", speaker3_voice: str = "Zephyr",
                                   speaker4_name: str = "", speaker4_voice: str = "Charon") -> Tuple[Dict[str, Any]]:
        """
        Generate multi-speaker speech from transcript
        """
        if not GOOGLE_AI_AVAILABLE:
            raise Exception("Google AI SDK not installed. Please install with: pip install google-genai")
        
        if not api_key.strip():
            raise Exception("API key is required. Get one from https://aistudio.google.com/")
        
        try:
            # Set up the API key
            os.environ['GOOGLE_API_KEY'] = api_key.strip()
            
            # Initialize the client
            client = genai.Client()
            
            # Extract voice names from "Voice (Style)" format
            def extract_voice_name(voice):
                return voice.split(' (')[0] if ' (' in voice else voice
            
            # Build speaker configurations
            speaker_configs = [
                types.SpeakerVoiceConfig(
                    speaker=speaker1_name,
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=extract_voice_name(speaker1_voice),
                        )
                    )
                ),
                types.SpeakerVoiceConfig(
                    speaker=speaker2_name,
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=extract_voice_name(speaker2_voice),
                        )
                    )
                ),
            ]
            
            # Add optional speakers
            if speaker3_name.strip():
                speaker_configs.append(
                    types.SpeakerVoiceConfig(
                        speaker=speaker3_name,
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=extract_voice_name(speaker3_voice),
                            )
                        )
                    )
                )
            
            if speaker4_name.strip():
                speaker_configs.append(
                    types.SpeakerVoiceConfig(
                        speaker=speaker4_name,
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=extract_voice_name(speaker4_voice),
                            )
                        )
                    )
                )
            
            # Generate the speech
            response = client.models.generate_content(
                model=model,
                contents=transcript,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                            speaker_voice_configs=speaker_configs
                        )
                    )
                )
            )
            
            # Extract audio data
            audio_data = response.candidates[0].content.parts[0].inline_data.data
            
            # Convert to audio format compatible with ComfyUI
            audio_dict = GoogleAIStudioTTSNode()._convert_audio_data(audio_data)
            
            return (audio_dict,)
            
        except Exception as e:
            raise Exception(f"Multi-speaker TTS generation failed: {str(e)}")


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GoogleAIStudioTTS": GoogleAIStudioTTSNode,
    "GoogleAIStudioMultiSpeakerTTS": GoogleAIStudioMultiSpeakerTTSNode,
    "GoogleAIStudioTextGen": GoogleAIStudioTextGenNode,
    "GoogleAIStudioImageGen": GoogleAIStudioImageGenNode,
    "GeminiPromptSplitter": GeminiPromptSplitter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GoogleAIStudioTTS": "Google AI Studio TTS",
    "GoogleAIStudioMultiSpeakerTTS": "Google AI Studio Multi-Speaker TTS",
    "GoogleAIStudioTextGen": "Google AI Studio Text Generator",
    "GoogleAIStudioImageGen": "Google AI Studio Image Generator",
    "GeminiPromptSplitter": "Gemini Prompt Splitter"
} 
