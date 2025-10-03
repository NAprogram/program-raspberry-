# lights.py
from __future__ import annotations
import asyncio
import requests

class BaseLightController:
    def turn_on(self): ...
    def turn_off(self): ...

class DummyLightController(BaseLightController):
    def turn_on(self):
        print("[LIGHT] (dummy) ON")
    def turn_off(self):
        print("[LIGHT] (dummy) OFF")

class HomeAssistantLightController(BaseLightController):
    def __init__(self, base_url: str, token: str, entity_id: str):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.entity_id = entity_id

    def _call(self, service: str):
        url = f"{self.base_url}/api/services/light/{service}"
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        body = {"entity_id": self.entity_id}
        r = requests.post(url, json=body, headers=headers, timeout=5)
        r.raise_for_status()

    def turn_on(self):
        self._call("turn_on")

    def turn_off(self):
        self._call("turn_off")

class KasaLightController(BaseLightController):
    """Control a TP-Link Kasa device directly on your LAN."""
    def __init__(self, host: str):
        self.host = host
        try:
            import kasa  # lazy import
            self.kasa = kasa
        except Exception as e:
            raise RuntimeError("Install python-kasa: pip install python-kasa") from e

    async def _with_device(self, coro):
        dev = self.kasa.SmartDevice(self.host)
        await dev.update()
        res = await coro(dev)
        await dev.update()
        return res

    def turn_on(self):
        asyncio.run(self._with_device(lambda d: d.turn_on()))

    def turn_off(self):
        asyncio.run(self._with_device(lambda d: d.turn_off()))

def make_light_controller(mode: str, cfg: dict) -> BaseLightController:
    mode = (mode or "dummy").lower()
    if mode == "homeassistant":
        return HomeAssistantLightController(
            cfg["homeassistant"]["base_url"],
            cfg["homeassistant"]["token"],
            cfg["homeassistant"]["entity_id"],
        )
    if mode == "kasa":
        return KasaLightController(cfg["kasa"]["host"])
    return DummyLightController()
