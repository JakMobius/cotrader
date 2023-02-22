import React from "react";
import { useEffect, useRef } from "react";
import { AppState, useApp } from "src/logic/app-state";
import { Candle, CandleChunk } from "src/logic/candle";
import EventEmitter from "src/utils/event-emitter";
import { padTime } from "src/utils/utils";

const Chart: React.FC = () => {
  const app = useApp()
  const chartRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!chartRef.current) return;
    const chart = new ChartWrapper(app, chartRef.current);

    return () => {
      chart.destroy();
    };
  }, []);

  return <div ref={chartRef} style={{ width: "100%", height: "100%" }} />;
}

export default Chart

class ChartWrapper implements ChartDatasource {

  app: AppState
  element: HTMLElement

  private drawer = new ChartDrawer(this)
  private leftChunkBoundary: number = 0
  private rightChunkBoundary: number = 0
  private visibleChunks = new Set<Candle[]>()
  private predictedChunks = new Set<Candle[]>()

  private hasPosition: boolean = false

  private destructor: (() => void) | null = null

  constructor(app: AppState, element: HTMLElement) {
    this.app = app
    this.element = element
    this.element.classList.add("chart")
    this.element.appendChild(this.drawer.canvas)

    this.destructor = this.setupEvents()

    this.loadInitialData()
  }

  getCandles(): Set<Candle[]> {
    return this.visibleChunks
  }

  getPredictedCandles(): Set<Candle[]> {
    return this.predictedChunks
  }

  getCandleLength(): number {
    return this.app.candleStore.intervalMs
  }

  private loadInitialData() {
    this.snapToLimit()
  }

  private chunkShouldBeDrawn(chunk: CandleChunk) {
    const chunkLength = this.app.candleStore.getChunkLength()
    const leftmostChunkTimestamp = this.leftChunkBoundary - chunkLength
    const rightmostChunkTimestamp = this.rightChunkBoundary + chunkLength

    if (chunk.timestamp < leftmostChunkTimestamp) return false
    if (chunk.timestamp > rightmostChunkTimestamp) return false

    return true
  }

  private candleVisible(candle: Candle) {
    const candleLength = this.app.candleStore.intervalMs
    const leftBoundary = this.drawer.boundaries.x0 - candleLength / 2
    const rightBoundary = this.drawer.boundaries.x1 + candleLength / 2

    if (candle.timestamp < leftBoundary) return false
    if (candle.timestamp > rightBoundary) return false

    return true
  }

  private addChunk(chunk: CandleChunk) {
    if (!chunk.candles.length) {
      return
    }

    let limit = this.app.dateLimit.get().getTime()

    if (chunk.candles[chunk.candles.length - 1].timestamp <= limit) {
      this.visibleChunks.add(chunk.candles)
      return
    }

    let candles = []
    for (let candle of chunk.candles) {
      if (candle.timestamp <= limit) {
        candles.push(candle)
      } else {
        break
      }
    }

    if (candles.length === 0) {
      return
    }
    this.visibleChunks.add(candles)
  }

  private maybeAddChunk(chunk: CandleChunk) {
    if (!this.chunkShouldBeDrawn(chunk)) {
      return
    }

    this.addChunk(chunk)
    if (!this.hasPosition) this.scaleToFit()
    this.drawer.setNeedsRedraw()
  }

  private reloadChunks() {
    this.visibleChunks.clear()
    this.drawer.setNeedsRedraw()

    let chunks = this.app.candleStore.getChunks(this.leftChunkBoundary, this.rightChunkBoundary, true)

    for (let chunk of chunks) {
      if (this.chunkShouldBeDrawn(chunk)) {
        this.addChunk(chunk)
      }
    }

    if (!this.hasPosition) this.scaleToFit()
    this.drawer.setNeedsRedraw()
  }

  private scaleToFit() {
    let y0 = Infinity
    let y1 = -Infinity
    for (let chunk of this.visibleChunks) {
      for (let candle of chunk) {
        if (this.candleVisible(candle)) {
          y0 = Math.min(y0, candle.indicators.open, candle.indicators.high, candle.indicators.low)
          y1 = Math.max(y1, candle.indicators.open, candle.indicators.high, candle.indicators.low)
        }
      }
    }

    if (y0 == Infinity) {
      return
    } else {
      let y0Scaled = y1 + (y0 - y1) * 1.3
      let y1Scaled = y0 + (y1 - y0) * 1.3
      y0 = y0Scaled
      y1 = y1Scaled
    }

    this.drawer.boundaries.y0 = y0
    this.drawer.boundaries.y1 = y1
    this.drawer.setNeedsRedraw()

    this.hasPosition = true
  }

  private updateBoundaries() {
    let start = this.drawer.boundaries.x0
    let end = this.drawer.boundaries.x1

    let newLeft = this.app.candleStore.roundChunk(start)
    let newRight = this.app.candleStore.roundChunk(end + this.app.candleStore.getChunkLength() - 1)

    if (newLeft === this.leftChunkBoundary && newRight === this.rightChunkBoundary) {
      return
    }

    this.leftChunkBoundary = newLeft
    this.rightChunkBoundary = newRight

    this.reloadChunks()
  }

  private snapToLimit() {
    this.hasPosition = false
    let date = this.app.dateLimit.get().getTime()
    let window = 30 * this.app.candleStore.intervalMs

    this.drawer.boundaries.x0 = date - window
    this.drawer.boundaries.x1 = date + window

    this.leftChunkBoundary = 0
    this.rightChunkBoundary = 0

    this.updateBoundaries()
  }

  private setupEvents() {
    this.drawer.on("move", () => {
      this.updateBoundaries()
      this.app.snapChart.set(false)
    })

    const onNewChunk = (chunk: CandleChunk) => {
      this.maybeAddChunk(chunk)
    }

    const onChunkUpdate = (chunk: CandleChunk) => {
      if (this.chunkShouldBeDrawn(chunk)) {
        this.reloadChunks()
      }
    }

    const reloadChunks = () => {
      if (this.app.snapChart.get() || !this.app.onlineMode.get()) {
        this.snapToLimit()
      } else {
        this.reloadChunks()
      }
    }

    const reset = () => {
      this.loadInitialData()
    }

    const onNewPredictedChunk = (chunk: CandleChunk) => {
      this.predictedChunks.add(chunk.candles)
      this.drawer.setNeedsRedraw()
    }

    const onPredictionReset = () => {
      this.predictedChunks.clear()
      this.drawer.setNeedsRedraw()
    }


    const onKeydown = (e: KeyboardEvent) => {
      if (e.target !== document.body) return
      if (e.code === "ArrowLeft") {
        let position = { ...this.drawer.boundaries }
        let newTime = this.app.dateLimit.get().getTime() - this.app.candleStore.intervalMs
        this.app.dateLimit.set(new Date(newTime))
        this.drawer.boundaries = position
        this.app.onlineMode.set(false)
      }
      if (e.code === "ArrowRight") {
        let position = { ...this.drawer.boundaries }
        let newTime = this.app.dateLimit.get().getTime() + this.app.candleStore.intervalMs
        this.app.dateLimit.set(new Date(newTime))
        this.drawer.boundaries = position
        this.app.onlineMode.set(false)
      }
    }

    const onKeyup = (e: KeyboardEvent) => {
      if (e.target !== document.body) return

    }

    document.body.addEventListener("keydown", onKeydown)
    document.body.addEventListener("keyup", onKeyup);

    this.app.candleStore.on("new-chunk", onNewChunk)
    this.app.candleStore.off("chunk-purged", onChunkUpdate)
    this.app.candleStore.on("chunk-update", onChunkUpdate)
    this.app.dateLimit.on("set", reloadChunks)
    this.app.candleStore.on("reset", reloadChunks)
    this.app.symbol.on("set", reset)
    this.app.snapChart.on("set", reloadChunks)

    this.app.predictionCandleStore.on("new-chunk", onNewPredictedChunk)
    this.app.predictionCandleStore.on("reset", onPredictionReset)

    return () => {
      this.app.candleStore.off("new-chunk", onNewChunk)
      this.app.candleStore.off("chunk-purged", onChunkUpdate)
      this.app.candleStore.off("chunk-update", onChunkUpdate)
      this.app.dateLimit.off("set", reloadChunks)
      this.app.candleStore.off("reset", reloadChunks)
      this.app.predictionCandleStore.off("new-chunk", onNewPredictedChunk)
      this.app.predictionCandleStore.off("reset", onPredictionReset)
      this.app.symbol.off("set", reset)
      this.app.snapChart.off("set", reloadChunks)

      document.body.removeEventListener("keydown", onKeydown);
      document.body.removeEventListener("keyup", onKeyup);
    }
  }

  destroy(): void {
    if (this.destructor) {
      this.destructor()
    }
  }
}

interface ChartBoundaries {
  x0: number,
  x1: number,
  y0: number,
  y1: number
}

interface ChartDatasource {
  getPredictedCandles(): Set<Candle[]>
  getCandles(): Set<Candle[]>
  getCandleLength(): number
}

interface ChartGridType<T> {
  interval: number
  format: (input: T) => string
}

class ChartDrawer extends EventEmitter {
  canvas: HTMLCanvasElement
  boundaries: ChartBoundaries = {
    x0: 0,
    x1: 0,
    y0: 0,
    y1: 0
  }
  ctx: CanvasRenderingContext2D
  widthPx: number = 0
  heightPx: number = 0
  height: number = 0
  width: number = 0
  frame: number | null = null
  dragging: boolean = false
  datasource: ChartDatasource

  // Style
  textLeft: number = 5
  textBottom: number = 5
  textboxMargin: number = 5

  mouseX: number | null = null
  mouseY: number | null = null

  static months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  static dateFormats = {
    hms: (date: Date) => {
      return `${padTime(date.getHours())}:${padTime(date.getMinutes())}:${padTime(date.getSeconds())}`;
    },
    hm: (date: Date) => {
      return `${padTime(date.getHours())}:${padTime(date.getMinutes())}`;
    },
    dm: (date: Date) => {
      return `${ChartDrawer.months[date.getMonth()]} ${padTime(date.getDate())}`;
    },
    my: (date: Date) => {
      return `${ChartDrawer.months[date.getMonth()]}'${String(date.getFullYear()).slice(-2)}`;
    },
  }
  static dateFractions: ChartGridType<Date>[] = [
    { interval: 1000, format: this.dateFormats.hms },
    { interval: 1000 * 15, format: this.dateFormats.hms },
    { interval: 1000 * 60, format: this.dateFormats.hm },
    { interval: 1000 * 60 * 15, format: this.dateFormats.hm },
    { interval: 1000 * 60 * 60, format: this.dateFormats.hm },
    { interval: 1000 * 60 * 60 * 3, format: this.dateFormats.hm },
    { interval: 1000 * 60 * 60 * 24, format: this.dateFormats.dm }
  ]

  constructor(datasource: ChartDatasource) {
    super()
    this.canvas = document.createElement("canvas")
    this.canvas.style.width = "100%"
    this.canvas.style.height = "100%"
    this.canvas.style.position = "absolute"
    this.ctx = this.canvas.getContext("2d")!
    this.datasource = datasource

    this.setupMouse()

    new ResizeObserver(() => this.resize()).observe(this.canvas)
  }

  private setupMouse() {
    let oldX = 0
    let oldY = 0

    const onZoom = (scale: number, horizontal: boolean, x: number, y: number) => {
      // Convert mouse position to chart coordinates
      const chartX = this.boundaries.x0 + (x / this.width) * (this.boundaries.x1 - this.boundaries.x0);
      const chartY = this.boundaries.y0 + (1 - y / this.height) * (this.boundaries.y1 - this.boundaries.y0);

      if (horizontal) {
        this.boundaries.x0 = chartX + (this.boundaries.x0 - chartX) * scale;
        this.boundaries.x1 = chartX + (this.boundaries.x1 - chartX) * scale;
      }

      this.boundaries.y0 = chartY + (this.boundaries.y0 - chartY) * scale;
      this.boundaries.y1 = chartY + (this.boundaries.y1 - chartY) * scale;

      this.emit("move");
      this.setNeedsRedraw();
    }

    const onMove = (dx: number, dy: number) => {
      dx = this.toChartWidth(dx)
      dy = this.toChartHeight(dy)

      this.boundaries.x0 -= dx
      this.boundaries.x1 -= dx
      this.boundaries.y0 -= dy
      this.boundaries.y1 -= dy

      this.emit("move")
      this.setNeedsRedraw()
    }

    const onMouseMove = (event: MouseEvent) => {
      let dx = event.pageX - oldX
      let dy = event.pageY - oldY

      onMove(dx, dy)

      oldX = event.pageX
      oldY = event.pageY
    }

    this.canvas.addEventListener("mousedown", (event) => {
      this.dragging = true
      oldX = event.pageX
      oldY = event.pageY
      document.body.addEventListener("mousemove", onMouseMove)
    })

    this.canvas.addEventListener("mouseup", () => {
      this.dragging = false
      document.body.removeEventListener("mousemove", onMouseMove)
    })

    this.canvas.addEventListener("mousemove", (event) => {
      let rect = this.canvas.getBoundingClientRect()
      this.mouseX = event.pageX - rect.left
      this.mouseY = event.pageY - rect.top
      this.setNeedsRedraw()
    })

    this.canvas.addEventListener("mouseleave", () => {
      this.mouseX = null
      this.mouseY = null
      this.setNeedsRedraw()
    })

    this.canvas.addEventListener("wheel", (event) => {
      event.preventDefault();

      const zoomIntensity = 0.05;
      const rect = this.canvas.getBoundingClientRect();
      const mouseX = event.clientX - rect.left;
      const mouseY = event.clientY - rect.top;

      if (event.ctrlKey && event.deltaY) {
        const scale = event.deltaY < 0 ? 1 - zoomIntensity : 1 + zoomIntensity;
        onZoom(scale, !event.shiftKey, mouseX, mouseY);
        return;
      }

      if (event.deltaZ) {
        const scale = event.deltaZ < 0 ? 1 - zoomIntensity : 1 + zoomIntensity;
        onZoom(scale, !event.shiftKey, mouseX, mouseY);
      }

      if (event.deltaX || event.deltaY) {
        onMove(-event.deltaX, -event.deltaY);
      }
    }, { passive: false });
  }

  resize() {
    const ratio = window.devicePixelRatio ?? 1
    this.widthPx = this.canvas.clientWidth * ratio
    this.heightPx = this.canvas.clientHeight * ratio
    this.width = this.canvas.clientWidth
    this.height = this.canvas.clientHeight
    this.canvas.width = this.widthPx
    this.canvas.height = this.heightPx
    this.ctx.resetTransform()
    this.ctx.scale(ratio, ratio)
    this.setNeedsRedraw()
  }

  toScreenX(x: number) {
    return (x - this.boundaries.x0) / (this.boundaries.x1 - this.boundaries.x0) * this.width
  }

  toScreenY(y: number) {
    return (1 - (y - this.boundaries.y0) / (this.boundaries.y1 - this.boundaries.y0)) * this.height
  }

  toScreenWidth(x: number) {
    return x / (this.boundaries.x1 - this.boundaries.x0) * this.width
  }

  toChartX(x: number) {
    return x / this.width * (this.boundaries.x1 - this.boundaries.x0) + this.boundaries.x0
  }

  toChartY(y: number) {
    return (1 - y / this.height) * (this.boundaries.y1 - this.boundaries.y0) + this.boundaries.y0
  }

  toChartWidth(x: number) {
    return x / this.width * (this.boundaries.x1 - this.boundaries.x0)
  }

  toChartHeight(y: number) {
    return -y / this.height * (this.boundaries.y1 - this.boundaries.y0)
  }

  private drawCandle(candle: Candle, predicted: boolean = false) {
    let length = this.datasource.getCandleLength()
    if (candle.timestamp > this.boundaries.x1 + length / 2) return
    if (candle.timestamp < this.boundaries.x0 - length / 2) return

    let timestamp = this.toScreenX(candle.timestamp)
    let o = this.toScreenY(candle.indicators.open)
    let h = this.toScreenY(candle.indicators.high)
    let l = this.toScreenY(candle.indicators.low)
    let c = this.toScreenY(candle.indicators.close)
    let v = this.toScreenY(candle.indicators.value)

    if (predicted) {
      let color = "rgb(188, 214, 168)"
      if (o < c) color = "rgb(215, 183, 183)"

      this.ctx.fillStyle = color
      this.ctx.strokeStyle = color
    } else {
      let color = "rgb(80, 179, 1)"
      if (o < c) color = "rgb(179, 1, 0)"

      this.ctx.fillStyle = color
      this.ctx.strokeStyle = color
    }

    this.ctx.beginPath()
    this.ctx.moveTo(timestamp, h)
    this.ctx.lineTo(timestamp, l)
    this.ctx.stroke()

    let candleWidth = this.toScreenWidth(length * 0.9)
    let halfCandleWidth = (candleWidth * 0.9) * 0.5

    this.ctx.fillRect(timestamp - halfCandleWidth, o, candleWidth, c - o)
  }

  private drawCandles() {
    for (let chunk of this.datasource.getCandles()) {
      for (let candle of chunk) {
        this.drawCandle(candle)
      }
    }
  }

  private drawPredictedCandles() {
    for (let chunk of this.datasource.getPredictedCandles()) {
      for (let candle of chunk) {
        this.drawCandle(candle, true)
      }
    }
  }

  private findSuitableInterval<T>(intervals: ChartGridType<T>[], target: number) {
    for (let i = 1; i < intervals.length; i++) {
      let small = intervals[i - 1]
      let large = intervals[i]
      if (small.interval <= target && large.interval > target) {
        return {
          small, large, factor: (target - small.interval) / (large.interval - small.interval)
        }
      }
    }

    if (intervals[0].interval > target) {
      return {
        large: intervals[0],
        small: null,
        factor: 1
      }
    }

    return {
      large: intervals[intervals.length - 1],
      small: null,
      factor: 1
    }
  }

  private getGridColor(factor: number) {
    return `rgba(220, 220, 220, ${factor})`
  }

  private getCrosshairColor() {
    return `rgb(180, 180, 180)`
  }

  private getGridTextColor() {
    return "rgb(128, 128, 128)"
  }

  private getCrosshairTextColor() {
    return "rgb(64, 64, 64)"
  }

  private getFontStyle() {
    return "14px monospace"
  }

  private drawXGrids() {
    const meaningfulDistance = 100 // pixels
    const visibleInterval = this.boundaries.x1 - this.boundaries.x0
    const meaningfulTimestep = visibleInterval / this.width * meaningfulDistance

    let { small, large, factor } = this.findSuitableInterval(ChartDrawer.dateFractions, meaningfulTimestep)

    if (small) this.drawXGrid(small, 1 - factor)
    if (large) this.drawXGrid(large, 1)

    return large
  }

  private drawXGrid(grid: ChartGridType<Date>, factor: number) {
    let start = Math.floor(this.boundaries.x0 / grid.interval) * grid.interval
    let end = Math.ceil(this.boundaries.x1 / grid.interval) * grid.interval

    this.ctx.beginPath()
    this.ctx.strokeStyle = this.getGridColor(factor)
    for (let ts = start; ts < end; ts += grid.interval) {
      let x = this.toScreenX(ts)
      this.ctx.moveTo(x, 0)
      this.ctx.lineTo(x, this.height)
    }
    this.ctx.stroke()
  }

  private drawXTexts(grid: ChartGridType<Date>) {
    let start = Math.floor(this.boundaries.x0 / grid.interval) * grid.interval
    let end = Math.ceil(this.boundaries.x1 / grid.interval) * grid.interval

    for (let ts = start; ts < end; ts += grid.interval) {
      this.drawXText(this.toScreenX(ts), grid.format(new Date(ts)))
    }
  }

  private drawBox(x: number, y: number, width: number, height: number, bold: boolean) {

    this.ctx.beginPath()
    this.ctx.moveTo(x, y)
    this.ctx.lineTo(x + width, y)
    this.ctx.lineTo(x + width, y + height)
    this.ctx.lineTo(x, y + height)
    this.ctx.closePath()
    this.ctx.fill()
    if (bold) {
      this.ctx.strokeStyle = "black"
      this.ctx.stroke()
    }
  }

  private drawXText(x: number, text: string, bold: boolean = false) {
    this.ctx.font = this.getFontStyle()
    this.ctx.textBaseline = "bottom"
    this.ctx.textAlign = "center"
    let textSize = this.ctx.measureText(text)
    let fontHeight = textSize.fontBoundingBoxAscent + textSize.fontBoundingBoxDescent

    this.ctx.fillStyle = "white"

    this.drawBox(
      x - textSize.width / 2 - this.textboxMargin,
      this.height - this.textBottom - fontHeight - this.textboxMargin * 2,
      textSize.width + this.textboxMargin * 2,
      fontHeight + this.textboxMargin * 2,
      bold
    )
    if (bold) {
      this.ctx.fillStyle = this.getCrosshairTextColor()
    } else {
      this.ctx.fillStyle = this.getGridTextColor()
    }
    this.ctx.fillText(
      text, x, this.height - this.textboxMargin - this.textBottom
    )
  }

  private drawYGrids() {
    const meaningfulDistance = 100 // pixels
    const visibleInterval = this.boundaries.y1 - this.boundaries.y0
    const meaningfulStep = visibleInterval / this.width * meaningfulDistance

    let log10 = Math.log10(meaningfulStep)
    let base = Math.pow(10, Math.floor(log10))
    let steps = [
      base,
      base * 2,
      base * 10
    ].map(step => {
      let decimals = Math.max(0, Math.ceil(-Math.log10(step)))
      return {
        interval: step,
        format: (input: number) => input.toFixed(decimals)
      } as ChartGridType<number>
    })

    let { small, large, factor } = this.findSuitableInterval(steps, meaningfulStep)

    if (small) this.drawYGrid(small, 1 - factor)
    if (large) this.drawYGrid(large, 1)

    return large
  }

  private drawYGrid(grid: ChartGridType<number>, factor: number) {
    let start = Math.floor(this.boundaries.y0 / grid.interval) * grid.interval
    let end = Math.ceil(this.boundaries.y1 / grid.interval) * grid.interval

    this.ctx.beginPath()
    this.ctx.strokeStyle = this.getGridColor(factor)
    for (let val = start; val < end; val += grid.interval) {
      let y = this.toScreenY(val)
      this.ctx.moveTo(0, y)
      this.ctx.lineTo(this.width, y)
    }
    this.ctx.stroke()
  }

  private drawYTexts(grid: ChartGridType<number>) {
    let start = Math.floor(this.boundaries.y0 / grid.interval) * grid.interval
    let end = Math.ceil(this.boundaries.y1 / grid.interval) * grid.interval

    for (let val = start; val < end; val += grid.interval) {
      this.drawYText(this.toScreenY(val), grid.format(val))
    }
  }

  private drawYText(y: number, text: string, bold: boolean = false) {
    this.ctx.font = this.getFontStyle()
    this.ctx.textBaseline = "middle"
    this.ctx.textAlign = "left"
    let textSize = this.ctx.measureText(text)
    let fontHeight = textSize.fontBoundingBoxAscent + textSize.fontBoundingBoxDescent

    this.ctx.fillStyle = "white"
    this.drawBox(
      this.textLeft,
      y - this.textboxMargin - fontHeight / 2,
      textSize.width + this.textboxMargin * 2,
      fontHeight + this.textboxMargin * 2,
      bold
    )
    if (bold) {
      this.ctx.fillStyle = this.getCrosshairTextColor()
    } else {
      this.ctx.fillStyle = this.getGridTextColor()
    }
    this.ctx.fillText(
      text, this.textLeft + this.textboxMargin, y
    )
  }

  private drawHoverCrosshair(xGrid: ChartGridType<Date>, yGrid: ChartGridType<number>) {
    if (this.dragging || !this.mouseX || !this.mouseY) return

    if (this.boundaries.x0 === this.boundaries.x1) return
    if (this.boundaries.y0 === this.boundaries.y1) return

    this.ctx.strokeStyle = this.getCrosshairColor()
    this.ctx.setLineDash([5, 5])
    this.ctx.beginPath()
    this.ctx.moveTo(this.mouseX, 0)
    this.ctx.lineTo(this.mouseX, this.height)
    this.ctx.moveTo(0, this.mouseY)
    this.ctx.lineTo(this.width, this.mouseY)
    this.ctx.stroke()
    this.ctx.setLineDash([])

    let x = this.toChartX(this.mouseX)
    let y = this.toChartY(this.mouseY)

    if (xGrid) this.drawXText(this.mouseX, xGrid.format(new Date(x)), true)
    if (yGrid) this.drawYText(this.mouseY, yGrid.format(y), true)
  }

  private draw() {
    this.frame = null
    this.ctx.clearRect(0, 0, this.widthPx, this.heightPx)

    let xGrid = this.drawXGrids()
    let yGrid = this.drawYGrids()
    this.drawCandles()
    this.drawPredictedCandles()

    if (yGrid) this.drawYTexts(yGrid)
    if (xGrid) this.drawXTexts(xGrid)

    this.drawHoverCrosshair(xGrid, yGrid)
  }

  setNeedsRedraw() {
    if (this.frame) return
    this.frame = requestAnimationFrame(() => this.draw())
  }
}
