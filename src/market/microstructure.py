import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


class MicrostructureAnalyzer:
    def __init__(self):
        self.price_cache: dict[str, list[float]] = {}

    def analyze(
        self,
        orderbook: dict,
        price_history: list[dict],
        current_price: float,
        volume: float,
        event_id: str,
    ) -> dict:
        ob_signals = self._orderbook_signals(orderbook)
        flow_signals = self._flow_signals(price_history, current_price)
        crowd_signals = self._crowd_behavior(price_history, current_price, volume)
        smart_money = self._smart_money_detection(price_history, current_price, volume)

        self._update_cache(event_id, current_price)

        combined = (
            ob_signals["orderbook_signal"] * 0.3 +
            flow_signals["flow_signal"] * 0.25 +
            crowd_signals["crowd_signal"] * 0.25 +
            smart_money["smart_money_signal"] * 0.2
        )

        return {
            **ob_signals,
            **flow_signals,
            **crowd_signals,
            **smart_money,
            "combined_micro_signal": float(np.clip(combined, -1, 1)),
            "micro_confidence": self._signal_confidence(ob_signals, flow_signals, crowd_signals),
        }

    def _orderbook_signals(self, orderbook: dict) -> dict:
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        if not bids and not asks:
            return {
                "orderbook_signal": 0.0, "bid_wall": False, "ask_wall": False,
                "depth_imbalance": 0.0, "spread_signal": 0.0,
            }

        bid_sizes = [float(b.get("size", 0)) for b in bids[:20]]
        ask_sizes = [float(a.get("size", 0)) for a in asks[:20]]

        total_bid = sum(bid_sizes)
        total_ask = sum(ask_sizes)
        total = total_bid + total_ask

        imbalance = (total_bid - total_ask) / max(total, 1)

        avg_bid = np.mean(bid_sizes) if bid_sizes else 0
        avg_ask = np.mean(ask_sizes) if ask_sizes else 0
        bid_wall = any(s > avg_bid * 3 for s in bid_sizes[:5]) if avg_bid > 0 else False
        ask_wall = any(s > avg_ask * 3 for s in ask_sizes[:5]) if avg_ask > 0 else False

        best_bid = float(bids[0].get("price", 0)) if bids else 0
        best_ask = float(asks[0].get("price", 1)) if asks else 1
        spread = best_ask - best_bid
        spread_signal = -spread * 5

        signal = imbalance * 0.6
        if bid_wall:
            signal += 0.15
        if ask_wall:
            signal -= 0.15
        signal += spread_signal * 0.1

        return {
            "orderbook_signal": float(np.clip(signal, -1, 1)),
            "bid_wall": bid_wall,
            "ask_wall": ask_wall,
            "depth_imbalance": imbalance,
            "spread_signal": spread_signal,
        }

    def _flow_signals(self, price_history: list[dict], current_price: float) -> dict:
        prices = self._extract_prices(price_history)

        if len(prices) < 5:
            return {
                "flow_signal": 0.0, "price_acceleration": 0.0,
                "reversal_signal": 0.0, "breakout_signal": 0.0,
            }

        returns = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

        recent_returns = returns[-5:]
        acceleration = sum(recent_returns) - sum(returns[-10:-5]) if len(returns) >= 10 else 0

        if len(prices) >= 20:
            recent_high = max(prices[-10:])
            recent_low = min(prices[-10:])
            prev_high = max(prices[-20:-10])
            prev_low = min(prices[-20:-10])

            if current_price > prev_high:
                breakout = 0.5
            elif current_price < prev_low:
                breakout = -0.5
            else:
                breakout = 0.0
        else:
            breakout = 0.0

        if len(prices) >= 10:
            mean_price = np.mean(prices[-20:])
            std_price = np.std(prices[-20:]) if len(prices) >= 20 else np.std(prices)
            if std_price > 0:
                z_score = (current_price - mean_price) / std_price
                if z_score > 2:
                    reversal = -0.3
                elif z_score < -2:
                    reversal = 0.3
                else:
                    reversal = 0.0
            else:
                reversal = 0.0
        else:
            reversal = 0.0

        flow = (
            acceleration * 10 * 0.4 +
            breakout * 0.35 +
            reversal * 0.25
        )

        return {
            "flow_signal": float(np.clip(flow, -1, 1)),
            "price_acceleration": acceleration,
            "reversal_signal": reversal,
            "breakout_signal": breakout,
        }

    def _crowd_behavior(self, price_history: list[dict], current_price: float, volume: float) -> dict:
        prices = self._extract_prices(price_history)

        if len(prices) < 10:
            return {
                "crowd_signal": 0.0, "herding_score": 0.0,
                "contrarian_score": 0.0, "panic_score": 0.0,
            }

        returns = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        recent = returns[-5:]
        same_direction = sum(1 for r in recent if r > 0) / len(recent)

        if same_direction > 0.8:
            herding = 0.4
        elif same_direction < 0.2:
            herding = -0.4
        else:
            herding = 0.0

        if current_price > 0.85 or current_price < 0.15:
            extreme_move = abs(sum(recent))
            if extreme_move > 0.1:
                panic = -0.3 if sum(recent) > 0 else 0.3
            else:
                panic = 0.0
        else:
            panic = 0.0

        mean = np.mean(prices[-20:]) if len(prices) >= 20 else np.mean(prices)
        if abs(current_price - mean) > 0.15:
            contrarian = -(current_price - mean) * 0.5
        else:
            contrarian = 0.0

        crowd = herding * 0.4 + contrarian * 0.35 + panic * 0.25

        return {
            "crowd_signal": float(np.clip(crowd, -1, 1)),
            "herding_score": herding,
            "contrarian_score": float(np.clip(contrarian, -1, 1)),
            "panic_score": panic,
        }

    def _smart_money_detection(
        self, price_history: list[dict], current_price: float, volume: float,
    ) -> dict:
        prices = self._extract_prices(price_history)

        if len(prices) < 10:
            return {
                "smart_money_signal": 0.0, "large_move_detected": False,
                "volume_spike": False, "stealth_accumulation": False,
            }

        returns = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
        avg_move = np.mean(returns) if returns else 0
        std_move = np.std(returns) if len(returns) > 1 else 0

        recent_move = abs(prices[-1] - prices[-2]) if len(prices) >= 2 else 0
        large_move = recent_move > avg_move + 2 * std_move if std_move > 0 else False

        recent_vol = np.std(prices[-5:]) if len(prices) >= 5 else 0
        older_vol = np.std(prices[-15:-5]) if len(prices) >= 15 else np.std(prices)
        volume_spike = recent_vol > older_vol * 1.5 if older_vol > 0 else False

        if len(prices) >= 10:
            small_moves = [
                prices[i] - prices[i - 1]
                for i in range(max(1, len(prices) - 10), len(prices))
            ]
            consistent_small = (
                all(m > 0 for m in small_moves) or all(m < 0 for m in small_moves)
            )
            stealth = consistent_small and not large_move
        else:
            stealth = False

        signal = 0.0
        if large_move:
            direction = 1 if prices[-1] > prices[-2] else -1
            signal += direction * 0.3
        if stealth:
            direction = 1 if prices[-1] > prices[-5] else -1
            signal += direction * 0.4
        if volume_spike:
            signal *= 1.2

        return {
            "smart_money_signal": float(np.clip(signal, -1, 1)),
            "large_move_detected": large_move,
            "volume_spike": volume_spike,
            "stealth_accumulation": stealth,
        }

    def _update_cache(self, event_id: str, price: float):
        if event_id not in self.price_cache:
            self.price_cache[event_id] = []
        self.price_cache[event_id].append(price)
        if len(self.price_cache[event_id]) > 1000:
            self.price_cache[event_id] = self.price_cache[event_id][-500:]

    @staticmethod
    def _extract_prices(history: list[dict]) -> list[float]:
        prices = []
        for h in history:
            p = h.get("p") or h.get("price") or h.get("yes")
            if p is not None:
                prices.append(float(p))
        return prices

    @staticmethod
    def _signal_confidence(ob: dict, flow: dict, crowd: dict) -> float:
        signals = [
            abs(ob["orderbook_signal"]),
            abs(flow["flow_signal"]),
            abs(crowd["crowd_signal"]),
        ]
        avg_strength = np.mean(signals)
        same_sign = (
            (ob["orderbook_signal"] >= 0) == (flow["flow_signal"] >= 0) == (crowd["crowd_signal"] >= 0)
        )
        conf = avg_strength * 0.6
        if same_sign:
            conf += 0.2
        return float(np.clip(conf, 0, 1))
