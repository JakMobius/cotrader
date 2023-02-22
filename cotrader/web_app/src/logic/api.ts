import { Candle } from "./candle";
import { Predictor } from "./app-state";

export class API {
    async getPredictors(): Promise<Predictor[]> {
        const res = await fetch(`/api/get-predictors`);
        if (!res.ok) throw new Error("Failed to fetch candles");
        const json = await res.json();
        if (json.error) {
            throw new Error(json.error)
        }
        return json.result as Predictor[]
    }

    async getCandles(symbol: string, interval: string, start: number, end: number): Promise<Candle[]> {
        const params = new URLSearchParams({
            symbol,
            interval,
            start: start.toString(),
            end: end.toString(),
        });
        const res = await fetch(`/api/get-candles?${params.toString()}`);
        if (!res.ok) throw new Error("Failed to fetch candles");
        const json = await res.json();
        if (json.error) {
            throw new Error(json.error)
        }
        return json.result.map((compressed: any) => {
            return {
                timestamp: compressed[0],
                indicators: compressed[1]
            } as Candle
        })
    }

    getPredictionSocket() {
        const proto = window.location.protocol === "https:" ? "wss" : "ws"
        const socket = new WebSocket(`${proto}://${window.location.host}/predict`);
        return socket;
    }

    getCandleSocket(symbol: string, interval: string) {
        const params = new URLSearchParams({
            symbol,
            interval
        });
        const proto = window.location.protocol === "https:" ? "wss" : "ws"
        return new WebSocket(`${proto}://${window.location.host}/candles?${params.toString()}`);
    }
}
