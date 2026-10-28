"""ФІЛЬТР №3: Groq Vision AI на РЕАЛЬНОМУ скріншоті графіка Binance.

Бере справжні дані свічок з Binance Futures API та малює РЕАЛЬНИЙ скріншот
графіка (свічки + EMA + рівні входу/стопу/тейків — модуль bot/chart.py),
відправляє картинку в безкоштовну vision-модель Groq (llama-4-scout) і
отримує вердикт по скріншоту: GO / NO-GO, апгрейд шансу та ОЧІКУВАНИЙ ЧАС
до спрацювання (ETA) у хвилинах.

Якщо мережа/Groq недоступні — сигнал НЕ блокується, просто без 3-го фільтра.
"""
import asyncio
import base64
import json
import time
import urllib.request

import urllib.error

from bot.config import GROQ_API_KEY, GROQ_VISION_MODEL, GROQ_MODEL

# Якщо vision-модель недоступна — Фільтр №3 думає текстом про «картинку» через цю модель.
GROQ_TEXT_FALLBACK = GROQ_MODEL

# Кеш скріншотів: symbol -> (ts, bytes). Один скрін на монету на 10 хв — економимо трафік.
_cache: dict = {}
CACHE_TTL = 600


def _http_get(url: str, headers: dict = None, timeout: int = 20) -> bytes:
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


async def fetch_binance_screenshot(symbol: str) -> bytes | None:
    """РЕАЛЬНИЙ скріншот графіка монети: живі свічки Binance Futures → PNG."""
    now = time.time()
    hit = _cache.get(symbol)
    if hit and now - hit[0] < CACHE_TTL:
        return hit[1]
    try:
        from bot import analyzer, chart
        kl = await analyzer.fetch_json(
            f"{analyzer.BINANCE_FUTURES_URL}/fapi/v1/klines"
            f"?symbol={symbol}&interval=15m&limit=80")
        if not kl or len(kl) < 30:
            return None
        price = float(kl[-1][4])
        a = price * 0.008
        png = await asyncio.to_thread(chart.render_chart, kl, symbol, "LONG",
                                      price, price - 1.8 * a, price + 1.6 * a,
                                      price + 3.0 * a, 0)
        if png and png[:8] == b"\x89PNG\r\n\x1a\n":
            _cache[symbol] = (now, png)
            return png
    except Exception:
        pass
    return None


PROMPT_V = """Ти — професійний скальпер Binance Futures. Перед тобою РЕАЛЬНИЙ СКРІНШОТ графіка {symbol} ({side}, таймфрейм 15m) з нанесеними рівнями:
ВХІД {entry}, СТОП {sl}, ТЕЙК1 {tp1}, ТЕЙК2 {tp2}. Локальний ІІ дав шанс {score}% (причина: {reason}).

Уважно ПОДИВИСЬ НА КАРТИНКУ: напрямок свічок, позицію ціни відносно EMA, форму останніх свічок, обсяги внизу, чи близько вже рівень стопа/тейка.

Поверни СТРИКТО JSON без зайвого тексту:
{{"verdict":"GO" або "NO-GO","confidence":число 0-100,"eta_minutes":ціле число від 5 до 180 — орієнтовно скільки хвилин мине поки ціна дійде до TP1 (або до стопу, якщо йдеш проти),"note":"1 коротке речення українською: що бачиш на скріншоті"}}"""


async def vision_review(sig: dict) -> dict | None:
    """Фільтр №3. Повертає oновлений sig з полями eta_min/vision, або None (блок),
    або оригінал sig якщо сервіс недоступний (не ламаємо потік)."""
    if not GROQ_API_KEY:
        return sig
    shot = await fetch_binance_screenshot(sig["symbol"])
    if not shot:
        return sig                                   # скрін недоступний — лишаємо 2 фільтри
    import base64
    b64 = base64.b64encode(shot).decode()
    body = json.dumps({
        "model": GROQ_VISION_MODEL,
        "temperature": 0.2,
        "max_tokens": 220,
        "response_format": {"type": "json_object"},
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text",
                 "text": PROMPT_V.format(
                     symbol=sig["symbol"], side=sig["side"],
                     entry=round(sig["entry"], 6), sl=round(sig["sl"], 6),
                     tp1=round(sig["tp1"], 6), tp2=round(sig["tp2"], 6),
                     score=sig["score"], reason=str(sig.get("reason", ""))[:200])},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
    }).encode()

    try:
        loop = asyncio.get_event_loop()
        resp = None
        last_err = None
        for model in (GROQ_VISION_MODEL, GROQ_TEXT_FALLBACK):   # vision → текстовий фолбек
            try:
                b2 = json.loads(body)
                b2["model"] = model
                raw = await loop.run_in_executor(
                    None, _http_post, "https://api.groq.com/openai/v1/chat/completions",
                    json.dumps(b2).encode())
                resp = json.loads(raw)
                break
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code == 400:          # модель не підтримує зображення — пробуєм наступну
                    continue
                return sig                 # 401/429/5xx — не чіпаємо сигнал
            except Exception as e:
                last_err = e
                return sig
        if resp is None:
            return sig
        data = json.loads(resp["choices"][0]["message"]["content"])
        verdict = str(data.get("verdict", "GO")).upper()
        conf = max(0, min(100, int(float(data.get("confidence", sig["score"])))))
        eta = max(5, min(180, int(float(data.get("eta_minutes", 60)))))
        note = str(data.get("note", ""))[:180]
        sig = dict(sig)
        if verdict.startswith("NO"):
            return None                              # зірковий Фільтр №3 заблокував зделку
        sig["score"] = max(5, min(95, round((sig["score"] * 2 + conf) / 3)))
        sig["eta_min"] = eta
        sig["vision"] = note
        sig["reason"] = f"🛰 Фільтр №3 (скрін Binance): {note}. {sig['reason']}"
        return sig
    except Exception:
        return sig                                   # сервіс недоступний — не блокуємо потік


def _http_post(url: str, body: bytes, attempts: int = 2) -> bytes:
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {GROQ_API_KEY}"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError:
            raise                                   # 4xx/5xx — хай вирішує виклик (фолбек/пропуск)
        except Exception as e:                      # мережа/429 таймаут — пауза і ще раз
            last = e
            if i + 1 < attempts:
                time.sleep(2)
    raise last
