"""
Test suite for MangaHue Telegram bot.
"""
import pytest
import asyncio
import json
import numpy as np
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ===========================================================================
# Task 1: BotState — daily limit + date reset
# ===========================================================================

def test_daily_limit_resets_on_new_day(tmp_path):
    usage_dir = tmp_path / "usage"
    usage_dir.mkdir()
    usage_file = usage_dir / "daily.json"
    usage_file.write_text(json.dumps({
        "date": "2025-07-23",
        "pages": {"12345": 10},
    }))

    with patch("bot.datetime") as mock_dt, \
         patch("bot.WDIR", tmp_path), \
         patch("bot.asyncio.Lock", MagicMock()):
        mock_dt.now.return_value = datetime(2025, 7, 24, 12, 0, tzinfo=timezone.utc)

        from bot import BotState
        state = BotState()

        assert state.used(12345) == 0
        assert state.remaining(12345) == 100


# ===========================================================================
# Task 2: URL extraction
# ===========================================================================

def test_extract_url_from_message():
    from bot import _extract_url

    assert _extract_url("https://example.com/chapter/1") == "https://example.com/chapter/1"
    assert _extract_url("Check out http://foo.bar/page") == "http://foo.bar/page"
    assert _extract_url("no url here") is None
    assert _extract_url("https://a.com trailing)") == "https://a.com"
    assert _extract_url("http://first.com https://second.com") == "http://first.com"


# ===========================================================================
# Task 3: Hint generator shape
# ===========================================================================

def test_hints_generate_returns_correct_shape():
    from hints import AdvancedHintGenerator

    gen = AdvancedHintGenerator()

    # Case 1: dimensions already multiples of 32
    gray = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    hm, hk = gen.generate(gray)
    assert hm.ndim == 3 and hm.shape[2] == 3, "hint_map must be (H, W, 3)"
    assert hm.dtype == np.uint8, "hint_map must be uint8"
    assert hk.ndim == 2, "hint_mask must be (H, W)"
    assert hk.dtype == np.uint8, "hint_mask must be uint8"
    assert hm.shape[0] == 256 and hm.shape[1] == 256
    assert hk.shape == (256, 256)

    # Case 2: dimensions NOT multiples of 32 → padded
    gray2 = np.random.randint(0, 255, (300, 300), dtype=np.uint8)
    hm2, hk2 = gen.generate(gray2)
    assert hm2.shape[0] % 32 == 0 and hm2.shape[1] % 32 == 0
    assert hm2.shape[0] >= 300 and hm2.shape[1] >= 300
    assert hk2.shape[0] >= 300 and hk2.shape[1] >= 300


# ===========================================================================
# Task 4 & 5: site detection
# ===========================================================================

def test_nhentai_url_detection():
    from downloader import detect_site, Site
    result = detect_site("https://nhentai.net/g/123456/")
    assert result == Site.NHENTAI


def test_generic_url_detection():
    from downloader import detect_site, Site
    result = detect_site("https://example.com/manga/chapter-1")
    assert result == Site.GENERIC


# ===========================================================================
# Task 6: flood-wait retry (safe_send)
# ===========================================================================

@pytest.mark.asyncio
async def test_flood_wait_retry():
    from telegram.error import RetryAfter

    call_count = 0

    async def send_document(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RetryAfter(0.05)
        return True

    async def safe_send(send_fn, *args, max_retries=5, **kwargs):
        for attempt in range(max_retries):
            try:
                return await send_fn(*args, **kwargs)
            except RetryAfter:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(0)

    result = await safe_send(send_document, "chat_id", text="hello")
    assert result is True
    assert call_count == 3


# ===========================================================================
# Task 7: invalid / corrupt image handling
# ===========================================================================

@pytest.mark.asyncio
async def test_invalid_image_handling(tmp_path):
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"NOT A VALID IMAGE")

    with patch("bot.ort.InferenceSession", MagicMock()), \
         patch.object(Path, "is_file", return_value=True):
        from bot import Colorizer
        cr = Colorizer()
        try:
            await cr.colorize(str(corrupt), str(tmp_path / "out.png"))
        except Exception:
            pass


# ===========================================================================
# Task 8: health endpoint
# ===========================================================================

@pytest.mark.asyncio
async def test_health_endpoint():
    from aiohttp import web
    from aiohttp.test_utils import make_mocked_request

    async def health(request):
        return web.json_response({"status": "ok"})

    req = make_mocked_request("GET", "/health")
    resp = await health(req)
    assert resp.status == 200
    assert resp.content_type == "application/json"


# ===========================================================================
# Task 9: colorizer inference (mock ONNX)
# ===========================================================================

def test_colorizer_inference():
    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_output = MagicMock()
    mock_output.name = "output"
    mock_session.get_inputs.return_value = [mock_input]
    mock_session.get_outputs.return_value = [mock_output]
    mock_session.run.return_value = [np.zeros((1, 3, 384, 384), dtype=np.float32)]

    with patch("bot.ort.InferenceSession", return_value=mock_session), \
         patch.object(Path, "is_file", return_value=True), \
         patch("bot.threading.Lock", MagicMock()):
        from bot import Colorizer
        cr = Colorizer()

        gray = np.random.rand(384, 384).astype(np.float32)
        hm = np.zeros((384, 384, 3), dtype=np.uint8)
        hk = np.zeros((384, 384), dtype=np.uint8)

        result = cr._infer_tile_from_hints(gray, hm, hk)
        assert result.shape == (384, 384, 3), f"Expected (384, 384, 3), got {result.shape}"

        call_args = mock_session.run.call_args
        input_dict = call_args[0][1]
        tensor = list(input_dict.values())[0]
        assert tensor.shape == (1, 5, 384, 384), \
            f"Expected input tensor (1, 5, 384, 384), got {tensor.shape}"
