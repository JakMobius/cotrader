import { useEffect, useState } from "react"
import EventEmitter from "../utils/event-emitter"
import { CandleStore, DownloaderCandleStore, PredictionCandleStore } from "./candle"
import React from "react"
import { API } from "./api"
import { intervalToMs } from "src/utils/utils"

export interface Predictor {
    id: string
    name?: string
    real?: boolean
}

class Value<T> extends EventEmitter {
    value: T

    constructor(initialValue: T) {
        super()
        this.value = initialValue
    }

    get() {
        return this.value
    }

    set(newValue: T) {
        if(this.value === newValue) return
        this.value = newValue
        this.emit("set", newValue)
    }

    use() {
        const [value, setValue] = useState(this.value);

        useEffect(() => {
            const handleChange = () => setValue(this.value);
            this.on('set', handleChange);
            return () => this.off('set', handleChange);
        }, []);

        return value;
    }
}

export class AppState extends EventEmitter {
    symbol = new Value<string>("BTCUSDT")
    interval = new Value<string>("5m")

    onlineMode = new Value<boolean>(true)
    snapChart = new Value<boolean>(true)
    dateLimit = new Value<Date>(new Date())
    activePredictor = new Value<Predictor | null>(null)
    predictionCandleStore = new PredictionCandleStore()

    api = new API()
    candleStore = new DownloaderCandleStore(this.api, this.symbol.get(), this.interval.get())

    constructor() {
        super()

        this.dateLimit.on("set", () => {
            let intervalMs = this.candleStore.intervalMs
            let ts = (Math.floor(this.dateLimit.get().getTime() / intervalMs)) * intervalMs

            if (this.candleStore.maxTimestamp !== null) {
                ts = Math.min(this.candleStore.maxTimestamp, ts)
            }

            if (ts !== this.dateLimit.get().getTime()) {
                this.dateLimit.set(new Date(ts))
            }
        })

        this.dateLimit.set(this.dateLimit.get())

        this.candleStore.on("max-timestamp-moved", () => {
            if (this.onlineMode.get()) {
                this.dateLimit.set(new Date(this.candleStore.maxTimestamp!))
            }
        })

        this.onlineMode.on("set", () => {
            if(!this.onlineMode.get()) return

            if(this.candleStore.maxTimestamp === null) {
                let intervalMs = this.candleStore.intervalMs
                let ts = (Math.floor(Date.now() / intervalMs)) * intervalMs

                this.dateLimit.set(new Date(ts))
            } else {
                this.dateLimit.set(new Date(this.candleStore.maxTimestamp!))
            }
        })

        this.interval.on("set", () => {
            this.dateLimit.set(this.dateLimit.get())
        })

        this.symbol.on("set", () => this.resetStore())
        this.interval.on("set", () => this.resetStore())

        this.activePredictor.on("set", () => this.updatePredictions())
        this.interval.on("set", () => this.updatePredictions())
        this.symbol.on("set", () => this.updatePredictions())
        this.dateLimit.on("set", () => this.updatePredictions())

        this.candleStore.connect()
    }

    updatePredictions() {
        let intervalMs = intervalToMs(this.interval.get())
        let dateLimit = this.dateLimit.get().getTime() + intervalMs
        let predictor = this.activePredictor.get()

        if (!predictor) {
            this.predictionCandleStore.clear()
            return
        }

        this.predictionCandleStore.reset(
            this.api,
            predictor.id,
            this.symbol.get(),
            this.interval.get(),
            dateLimit
        )
    }

    private resetStore() {
        this.candleStore.reset(this.symbol.get(), this.interval.get())
    }
}

export const AppContext = React.createContext<AppState | undefined>(undefined);

export function useApp() {
    const context = React.useContext(AppContext);
    if (!context) {
        throw new Error("useApp must be used within an AppContext.Provider");
    }
    return context;
}
