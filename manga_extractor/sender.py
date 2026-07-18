"""Telegram batch image sender - sends pages 10 at a time in order."""
import json
import io
import logging
import asyncio
from typing import Optional, Callable
import httpx
from .utils import chunked

logger = logging.getLogger(__name__)

_TG_API = "https://api.telegram.org"


class TelegramSender:
    """Send manga pages to a Telegram chat in ordered batches.

    Usage:
        sender = TelegramSender(bot_token)
        async with sender:
            await sender.send_batch(chat_id, image_urls)
    """

    def __init__(
        self,
        bot_token: str,
        batch_size: int = 10,
        delay: float = 1.2,
        max_retries: int = 3,
        timeout: float = 30,
    ):
        self._token = bot_token.strip()
        self._batch_size = min(batch_size, 10)
        self._delay = delay
        self._max_retries = max_retries
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout),
        )
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if not self._client:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self._timeout))
        return self._client

    def _url(self, method: str) -> str:
        return f"{_TG_API}/bot{self._token}/{method}"

    async def send_message(self, chat_id: int | str, text: str) -> bool:
        try:
            r = await self.client.post(self._url("sendMessage"), json={
                "chat_id": str(chat_id), "text": text,
            })
            return r.json().get("ok", False)
        except Exception:
            return False

    async def send_batch(
        self,
        chat_id: int | str,
        urls: list[str],
        caption: str = "",
        progress: Optional[Callable] = None,
    ) -> int:
        """Download images and send them to chat in batches of 10.

        Returns total pages successfully sent.
        """
        total = len(urls)
        sent = 0

        for batch_idx, batch_urls in enumerate(chunked(urls, self._batch_size)):
            # Download all images in this batch
            images: list[tuple[str, bytes]] = []
            for i, url in enumerate(batch_urls):
                data = await self._download(url)
                if data:
                    name = url.rsplit("/", 1)[-1].split("?")[0] or f"{batch_idx * self._batch_size + i + 1}.jpg"
                    images.append((name, data))

            if not images:
                continue

            # Build media array
            media = []
            for j, (name, _) in enumerate(images):
                item = {"type": "photo", "media": f"attach://file_{j}"}
                media.append(item)

            # Multipart: chat_id + media JSON + photos
            boundary = f"mangabot{asyncio.get_event_loop().time()}"
            body = bytearray()
            chat_id_str = str(chat_id)

            body += f"--{boundary}\r\n".encode()
            body += f'Content-Disposition: form-data; name="chat_id"\r\n\r\n'.encode()
            body += f"{chat_id_str}\r\n".encode()

            body += f"--{boundary}\r\n".encode()
            body += f'Content-Disposition: form-data; name="media"\r\n'.encode()
            body += b"Content-Type: application/json\r\n\r\n"
            body += json.dumps(media).encode()
            body += b"\r\n"

            for j, (name, data) in enumerate(images):
                body += f"--{boundary}\r\n".encode()
                body += f'Content-Disposition: form-data; name="file_{j}"; filename="{name}"\r\n'.encode()
                body += b"Content-Type: application/octet-stream\r\n\r\n"
                body += data
                body += b"\r\n"

            body += f"--{boundary}--\r\n".encode()

            # Send
            ok = False
            for attempt in range(self._max_retries):
                try:
                    r = await self.client.post(
                        self._url("sendMediaGroup"),
                        content=bytes(body),
                        headers={
                            "Content-Type": f"multipart/form-data; boundary={boundary}",
                        },
                    )
                    resp = r.json()
                    if resp.get("ok"):
                        ok = True
                        break
                    desc = resp.get("description", "")
                    logger.warning(f"sendMediaGroup warn: {desc}")
                except Exception as e:
                    logger.error(f"sendMediaGroup try {attempt+1}: {e}")
                    if attempt < self._max_retries - 1:
                        await asyncio.sleep(1.0 * (attempt + 1))
                    else:
                        logger.error(f"Failed completely for batch {batch_idx}")

            if ok:
                sent += len(images)
                if progress:
                    progress(sent, total)

            await asyncio.sleep(self._delay)

        return sent

    async def _download(self, url: str) -> bytes | None:
        for attempt in range(self._max_retries):
            try:
                r = await self.client.get(url)
                if r.status_code == 200:
                    return r.content
                if r.status_code >= 500:
                    raise Exception(f"Server error {r.status_code}")
                return None  # 404 etc -> skip
            except Exception:
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))
        return None
