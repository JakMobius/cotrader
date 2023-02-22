
import { intervalToMs } from "../utils/utils"
import EventEmitter from "src/utils/event-emitter"
import { API } from "./api"

export type Candle = {
    timestamp: number
    indicators: { [key: string]: number }
}

export class CandleChunk {
    timestamp: number
    candles: Candle[]
    constructor(candles: Candle[]) {
        this.timestamp = candles[0].timestamp
        this.candles = candles
    }
}

export interface AsyncCandleDatasource {
    getCandles(symbol: string, interval: string, from: number, to: number): Promise<Candle[]>
}

export interface CandleStore extends EventEmitter {
    interval: string | null
    intervalMs: number | null

    getChunkLength(): number
    roundChunk(atTimestamp: number): number
    getChunks(from: number, to: number): CandleChunk[]
}

export class PredictionCandleStore extends EventEmitter implements CandleStore {
    store = new BinaryTree<CandleChunk>()
    symbol: string | null = null
    interval: string | null = null
    intervalMs: number | null = null
    socket: WebSocket | null = null
    model: string | null = null
    dateLimit: number | null = null
    chunks: CandleChunk[] = []

    targetPredictionCount: number = 16
    finished: boolean = false

    savedData: Map<string, CandleChunk> = new Map()

    isSocketBusy: boolean = false
    interruptPrediction: boolean = false

    connect(api: API) {
        this.isSocketBusy = true
        this.socket = api.getPredictionSocket()
        this.socket.onmessage = (message: MessageEvent) => this.onMessage(message)
        this.socket.onerror = (error) => {
            console.error(error)
            this.socket = null
        }
        this.socket.onclose = () => {
            this.socket = null
        }
        this.socket.onopen = () => {
            this.onSocketFree()
        }
    }

    clear() {
        this.model = null
        this.finished = false
        this.chunks = []
        if (this.isSocketBusy) {
            this.interruptPrediction = true
        }
        this.emit("reset")
    }

    reset(api: API, model: string, symbol: string, interval: string, dateLimit: number) {
        if (this.model === model && this.symbol === symbol && this.interval === interval && this.dateLimit === dateLimit)
            return

        this.model = model
        this.dateLimit = dateLimit
        this.interval = interval
        this.symbol = symbol
        this.intervalMs = intervalToMs(this.interval)
        this.finished = false
        this.emit("reset")

        let configStr = this.getConfigStr()

        if (this.savedData.has(configStr)) {
            let savedChunk = this.savedData.get(configStr)!
            this.chunks = [savedChunk]
            this.finished = true
            this.emit("new-chunk", savedChunk)
            this.emit("finish")
            this.interruptPrediction = true
        } else {
            this.chunks = []
            if (!this.socket) {
                this.connect(api)
            }
            this.makeRequest()
        }
    }

    makeRequest() {
        if (!this.socket || !this.model) return
        if (this.isSocketBusy) {
            this.interruptPrediction = true
            return
        }
        this.isSocketBusy = true
        this.socket.send(JSON.stringify({
            model: this.model,
            symbol: this.symbol,
            interval: this.interval,
            start: this.dateLimit,
            predictions: this.targetPredictionCount
        }))
        this.interruptPrediction = false
    }

    getProgress() {
        if (this.finished) return 1.0
        return this.chunks.length / this.targetPredictionCount
    }

    hasDataFor(model: string) {
        let myModel = this.model
        this.model = model
        let str = this.getConfigStr()
        this.model = myModel
        return this.savedData.has(str)
    }

    getConfigStr() {
        return this.symbol + "_" + this.interval + "_" + this.model + "_" + this.dateLimit
    }

    onMessage(message: MessageEvent) {

        let json: any = null
        try {
            json = JSON.parse(message.data)
        } catch (e) {
            json = { "type": "error", "error": e }
        }

        if (json.type === "prediction") {
            if (this.interruptPrediction) {
                this.socket?.send(JSON.stringify({ "type": "stop" }))
            } else {
                let chunk = new CandleChunk([{
                    timestamp: json.candle[0],
                    indicators: json.candle[1],
                }])
                this.chunks.push(chunk)
                this.emit("new-chunk", chunk)
                this.socket?.send(JSON.stringify({ "type": "ack" }))
            }
        }
        if (json.type === "finish") {
            if (!this.interruptPrediction) {
                this.finished = true
                if (this.chunks.length === this.targetPredictionCount) {
                    let candles = []
                    for (let chunk of this.chunks) {
                        candles.push(...chunk.candles)
                    }
                    this.savedData.set(this.getConfigStr(), new CandleChunk(candles))
                }
                this.emit("finish")
            }
            this.onSocketFree()
        }
        if (json.type === "error") {
            console.error("Prediction error", json.error)
        }
    }

    onSocketFree() {
        this.isSocketBusy = false
        if (this.interruptPrediction) {
            this.makeRequest()
        }
    }

    getChunkLength() {
        if (this.intervalMs === null) throw new Error("Store not ready")
        return this.intervalMs
    }

    roundChunk(atTimestamp: number) {
        const scale = this.getChunkLength()
        return Math.floor(atTimestamp / scale) * scale
    }

    getChunks(from: number, to: number) {
        let result = []
        for (let chunk of this.chunks) {
            if (chunk.timestamp >= from && chunk.timestamp < to) {
                result.push(chunk)
            }
        }
        return result
    }
}

export class DownloaderCandleStore extends EventEmitter implements CandleStore {
    store = new BinaryTree<CandleChunk>()
    interval: string
    symbol: string
    intervalMs: number
    loadingChunks = new Set<number>()
    api: API
    chunkAlignment = 128

    maxTimestamp: number | null = null

    incompleteChunks = new Set<CandleChunk>()
    socket: WebSocket | null = null

    constructor(api: API, symbol: string, interval: string) {
        super()
        this.symbol = symbol
        this.interval = interval
        this.intervalMs = intervalToMs(this.interval)
        this.api = api
        this.reset(symbol, interval)
    }

    reset(symbol: string, interval: string) {
        if (this.symbol === symbol && this.interval === interval) {
            return
        }
        if (this.socket) {
            this.socket.close()
            this.socket = null
        }
        this.symbol = symbol
        this.interval = interval
        this.intervalMs = intervalToMs(this.interval)
        this.incompleteChunks.clear()
        // Create new store here, because racy requests might
        // fill the store with data for the old traiding pair
        this.store = new BinaryTree()
        this.loadingChunks.clear()
        this.emit("reset")
    }

    connect() {
        if (this.socket) return
        this.socket = this.api.getCandleSocket(this.symbol, this.interval)
        this.socket.onerror = (error) => {
            console.error(error)
            this.socket = null
            setTimeout(() => this.connect(), 1000)
        }
        this.socket.onclose = () => {
            this.socket = null
            setTimeout(() => this.connect(), 1000)
        }
        this.socket.onmessage = (message) => {
            this.onNewCandle(message)
        }
    }

    updateMaxTimestamp(timestamp: number) {
        if (this.maxTimestamp !== null && this.maxTimestamp > timestamp) return
        this.maxTimestamp = timestamp
        this.emit("max-timestamp-moved")
    }

    onNewCandle(message: MessageEvent) {
        let json = JSON.parse(message.data)
        if (json.type !== "candle") return
        let candle: Candle = {
            timestamp: json.candle[0],
            indicators: json.candle[1]
        }

        if (candle.timestamp % this.intervalMs !== 0) {
            console.warn(`[onNewCandle] Received unaligned timestamp: ${candle.timestamp}`)
        }

        let scale = this.chunkAlignment * this.intervalMs

        for (let chunk of this.incompleteChunks) {
            let lowBoundary = chunk.timestamp
            let highBoundary = lowBoundary + scale

            if (candle.timestamp >= highBoundary) {
                this.incompleteChunks.delete(chunk)
                this.emit("chunk-purged", chunk)
                this.downloadChunk(chunk.timestamp / scale)
                continue
            }

            if (candle.timestamp < lowBoundary) {
                console.warn("[onNewCandle] Adding candle to the past. You are that much out of sync?")
                continue
            }

            for (let existingCandle of chunk.candles) {
                if (candle.timestamp === existingCandle.timestamp) {
                    console.warn("[onNewCandle] The candle conflicts with an existing one. You are that much out of sync?")
                    return
                }
            }

            let expectedTimestamp = chunk.candles[chunk.candles.length - 1].timestamp + this.intervalMs

            if (candle.timestamp > expectedTimestamp) {
                let candleCount = (candle.timestamp - expectedTimestamp) / this.intervalMs
                console.warn(`[onNewCandle] Socket skipped ${candleCount} candles within a chunk: ${expectedTimestamp}...${candle.timestamp}. Re-downloading...`)
                this.downloadPartialChunk(expectedTimestamp, candle.timestamp, chunk)
            }

            chunk.candles.push(candle)
            this.updateMaxTimestamp(candle.timestamp)

            if (chunk.candles.length === this.chunkAlignment) {
                this.incompleteChunks.delete(chunk)
                this.store.insert(chunk)
            }

            this.emit("chunk-update", chunk)
            return
        }

        let newChunk = new CandleChunk([candle])
        this.incompleteChunks.add(newChunk)
        this.emit("new-chunk", newChunk)
    }

    downloadPartialChunk(from: number, to: number, chunk: CandleChunk) {
        let store = this.store
        this.api.getCandles(this.symbol, this.interval, from, to).then((candles) => {
            if (store !== this.store) return

            let newCandles = []
            let i = 0, j = 0;
            const orig = chunk.candles;
            while (i < orig.length && j < candles.length) {
                if (orig[i].timestamp === candles[j].timestamp) {
                    console.warn("[downloadPartialChunk] Downloaded partial chunk conflicts with existing data");
                    return;
                } else if (orig[i].timestamp < candles[j].timestamp) {
                    newCandles.push(orig[i]);
                    i++;
                } else {
                    newCandles.push(candles[j]);
                    j++;
                }
            }
            while (i < orig.length) {
                newCandles.push(orig[i++]);
            }
            while (j < candles.length) {
                newCandles.push(candles[j++]);
            }
            chunk.candles = newCandles;
            this.emit("chunk-update", chunk);
        }).catch(e => {
            console.error(e)
            this.emit("error", e)
        })
    }

    downloadChunk(index: number) {
        if (this.loadingChunks.has(index)) return

        const scale = this.intervalMs * this.chunkAlignment
        const from = scale * index
        const to = from + scale

        // Avoid downloading the chunks far in the future.
        for (let chunk of this.incompleteChunks) {
            if (chunk.timestamp <= from) return
        }

        this.loadingChunks.add(index)

        console.log(`[downloadChunk] Downloading chunk #${index}: ${new Date(from)}...${new Date(to)}`)

        const store = this.store

        this.api.getCandles(this.symbol, this.interval, from, to).then((candles) => {
            if (store !== this.store) {
                console.log(`[downloadChunk] Downloaded Chunk #${index} for old store, discarding`)
            }
            this.loadingChunks.delete(index)
            if (candles.length === 0) {
                console.log(`[downloadChunk] Chunk #${index} is empty`)
                return
            }

            let chunk = new CandleChunk(candles)
            if (candles.length !== this.chunkAlignment) {
                console.log(`[downloadChunk] Received incomplete chunk #{index} of length ${chunk.candles.length}`)
                this.incompleteChunks.add(chunk)
            } else {
                console.log(`[downloadChunk] Downloaded chunk #${index}`)
                store.insert(chunk)
            }
            this.updateMaxTimestamp(from + (candles.length - 1) * this.intervalMs)
            this.emit("new-chunk", chunk)
        }).catch(e => {
            this.loadingChunks.delete(index)
            console.error(e)
            this.emit("error", e)
        })
    }

    getChunkLength() {
        return this.intervalMs * this.chunkAlignment
    }

    roundChunk(atTimestamp: number) {
        const scale = this.getChunkLength()
        return Math.floor(atTimestamp / scale) * scale
    }

    getChunks(from: number, to: number, download: boolean = false) {
        const scale = this.getChunkLength()
        const fromChunk = Math.floor(from / scale)
        const toChunk = Math.floor((to + (scale - 1)) / scale)

        const chunks = this.store.query(fromChunk * scale, toChunk * scale)

        if (download) {
            for (let i = fromChunk, j = 0; i < toChunk; i++, j++) {
                if (j >= chunks.length) {
                    this.downloadChunk(i)
                    continue
                }
                while (chunks[j]?.timestamp !== i * scale && i < toChunk) {
                    this.downloadChunk(i)
                    i++
                }
            }
        }

        for (let chunk of this.incompleteChunks) {
            let chunkStart = chunk.timestamp
            let chunkEnd = chunkStart + this.intervalMs + this.chunkAlignment
            if (!(from > chunkEnd || to < chunkStart)) {
                chunks.push(chunk)
            }
        }

        return chunks
    }
}

export interface ITreeNode {
    timestamp: number
}

export class BinaryTree<T extends ITreeNode> {
    private root: BinaryTreeNode<T> | null = null;

    insert(element: T): void {
        this.root = this._insert(this.root, element);
    }

    query(start: number, end: number): T[] {
        const result: T[] = [];
        this._query(this.root, start, end, result);
        return result;
    }

    clear(): void {
        this.root = null;
    }

    private _insert(node: BinaryTreeNode<T> | null, element: T): BinaryTreeNode<T> {
        if (!node) return new BinaryTreeNode<T>(element);
        if (element.timestamp < node.element.timestamp) {
            node.left = this._insert(node.left, element);
        } else {
            node.right = this._insert(node.right, element);
        }
        return node;
    }

    private _query(node: BinaryTreeNode<T> | null, start: number, end: number, result: T[]): void {
        if (!node) return;
        if (node.element.timestamp > start) {
            this._query(node.left, start, end, result);
        }
        if (node.element.timestamp >= start && node.element.timestamp <= end) {
            result.push(node.element);
        }
        if (node.element.timestamp < end) {
            this._query(node.right, start, end, result);
        }
    }
}

class BinaryTreeNode<T> {
    element: T;
    left: BinaryTreeNode<T> | null = null;
    right: BinaryTreeNode<T> | null = null;

    constructor(element: T) {
        this.element = element;
    }
}
