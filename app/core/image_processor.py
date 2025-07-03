import asyncio
import base64
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import BytesIO
from typing import List, Optional, Dict
import tempfile
import aiohttp
import time
import gc
from PIL import Image
from loguru import logger

class ImageProcessor:
    def __init__(self, max_workers: int = 4, cache_size: int = 1000):
        # Use tempfile for macOS-efficient temporary file handling
        self.temp_dir = tempfile.TemporaryDirectory()
        self._session: Optional[aiohttp.ClientSession] = None
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._cache_size = cache_size
        self._last_cleanup = time.time()
        self._cleanup_interval = 3600  # 1 hour
        # Replace lru_cache with manual cache for better control
        self._hash_cache: Dict[str, str] = {}
        self._cache_access_times: Dict[str, float] = {}
        Image.MAX_IMAGE_PIXELS = 100000000  # Limit to 100 megapixels

    def _get_image_hash(self, image_url: str) -> str:
        """Get hash for image URL with manual caching that can be cleared."""
        # Check if already cached
        if image_url in self._hash_cache:
            self._cache_access_times[image_url] = time.time()
            return self._hash_cache[image_url]
        
        # Generate hash
        if image_url.startswith("data:"):
            _, encoded = image_url.split(",", 1)
            data = base64.b64decode(encoded)
        else:
            data = image_url.encode('utf-8')
        
        hash_value = hashlib.md5(data).hexdigest()
        
        # Add to cache with size management
        if len(self._hash_cache) >= self._cache_size:
            self._evict_oldest_cache_entries()
        
        self._hash_cache[image_url] = hash_value
        self._cache_access_times[image_url] = time.time()
        return hash_value

    def _evict_oldest_cache_entries(self):
        """Remove oldest 20% of cache entries to make room."""
        if not self._cache_access_times:
            return
            
        # Sort by access time and remove oldest 20%
        sorted_items = sorted(self._cache_access_times.items(), key=lambda x: x[1])
        to_remove = len(sorted_items) // 5  # Remove 20%
        
        for url, _ in sorted_items[:to_remove]:
            self._hash_cache.pop(url, None)
            self._cache_access_times.pop(url, None)
        
        # Force garbage collection after cache eviction
        gc.collect()

    def _resize_image_keep_aspect_ratio(self, image: Image.Image, max_size: int = 448) -> Image.Image:
        width, height = image.size
        if width <= max_size and height <= max_size:
            return image
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)
            
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _prepare_image_for_saving(self, image: Image.Image) -> Image.Image:
        if image.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'RGBA':
                background.paste(image, mask=image.split()[3])
            else:
                background.paste(image, mask=image.split()[1])
            return background
        elif image.mode != 'RGB':
            return image.convert('RGB')
        return image

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={"User-Agent": "mlx-server-OAI-compat/1.0"}
            )
        return self._session

    def _cleanup_old_files(self):
        current_time = time.time()
        if current_time - self._last_cleanup > self._cleanup_interval:
            try:
                for file in os.listdir(self.temp_dir.name):
                    file_path = os.path.join(self.temp_dir.name, file)
                    if os.path.getmtime(file_path) < current_time - self._cleanup_interval:
                        os.remove(file_path)
                self._last_cleanup = current_time
                # Also clean up cache periodically
                if len(self._hash_cache) > self._cache_size * 0.8:
                    self._evict_oldest_cache_entries()
                gc.collect()  # Force garbage collection after cleanup
            except Exception as e:
                logger.warning(f"Failed to clean up old files: {str(e)}")

    async def _process_single_image(self, image_url: str) -> str:
        image = None
        try:
            image_hash = self._get_image_hash(image_url)
            cached_path = os.path.join(self.temp_dir.name, f"{image_hash}.jpg")

            if os.path.exists(cached_path):
                logger.debug(f"Using cached image: {cached_path}")
                return cached_path

            if os.path.exists(image_url):
                # Read-only image loading for memory efficiency
                with Image.open(image_url, mode='r') as image:
                    image = self._resize_image_keep_aspect_ratio(image)
                    image = self._prepare_image_for_saving(image)
                    image.save(cached_path, 'JPEG', quality=100, optimize=True)
                return cached_path

            elif image_url.startswith("data:"):
                _, encoded = image_url.split(",", 1)
                estimated_size = len(encoded) * 3 / 4
                if estimated_size > 100 * 1024 * 1024:
                    raise ValueError("Base64-encoded image exceeds 100 MB")
                data = base64.b64decode(encoded)
                with Image.open(BytesIO(data), mode='r') as image:
                    image = self._resize_image_keep_aspect_ratio(image)
                    image = self._prepare_image_for_saving(image)
                    image.save(cached_path, 'JPEG', quality=100, optimize=True)
            else:
                session = await self._get_session()
                async with session.get(image_url) as response:
                    response.raise_for_status()
                    data = await response.read()
                    with Image.open(BytesIO(data), mode='r') as image:
                        image = self._resize_image_keep_aspect_ratio(image)
                        image = self._prepare_image_for_saving(image)
                        image.save(cached_path, 'JPEG', quality=100, optimize=True)

            self._cleanup_old_files()
            return cached_path

        except Exception as e:
            logger.error(f"Failed to process image: {str(e)}")
            raise ValueError(f"Failed to process image: {str(e)}")
        finally:
            # Ensure image object is closed to free memory
            if image:
                try:
                    image.close()
                except:
                    pass
            gc.collect()

    async def process_image_url(self, image_url: str) -> str:
        return await self._process_single_image(image_url)

    async def process_image_urls(self, image_urls: List[str]) -> List[str]:
        tasks = [self.process_image_url(url) for url in image_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Force garbage collection after batch processing
        gc.collect()
        return results

    def clear_cache(self):
        """Manually clear the hash cache to free memory."""
        self._hash_cache.clear()
        self._cache_access_times.clear()
        gc.collect()

    async def cleanup(self):
        if hasattr(self, '_cleaned') and self._cleaned:
            return
        self._cleaned = True
        try:
            # Clear caches before cleanup
            self.clear_cache()
            
            if self._session and not self._session.closed:
                await self._session.close()
        except Exception as e:
            logger.warning(f"Exception closing aiohttp session: {str(e)}")
        try:
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.warning(f"Exception shutting down executor: {str(e)}")
        try:
            self.temp_dir.cleanup()
        except Exception as e:
            logger.warning(f"Exception cleaning up temp_dir: {str(e)}")
        
        # Final garbage collection
        gc.collect()

    async def __aenter__(self):
        """Enter the async context manager."""
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Exit the async context manager and perform cleanup."""
        await self.cleanup()

    def __del__(self):
        # Async cleanup cannot be reliably performed in __del__
        # Please use 'async with ImageProcessor()' or call 'await cleanup()' explicitly.
        pass