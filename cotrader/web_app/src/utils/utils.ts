
const TIMES = new Map<string, number>([
    ["s", 1000],
    ["m", 60 * 1000],
    ["h", 60 * 60 * 1000],
    ["d", 24 * 60 * 60 * 1000],
    ["w", 7 * 24 * 60 * 60 * 1000],
    ["M", 30 * 24 * 60 * 60 * 1000],
    ["Y", 365 * 24 * 60 * 60 * 1000],
])

export function intervalToMs(interval: string) {
    let unit = interval.slice(-1)
    let count = parseInt(interval.slice(0, -1))

    if(!TIMES.has(unit) || isNaN(count)) {
        throw new Error("Invalid interval: " + interval)
    }

    return TIMES.get(unit)! * count
}

export function padTime(n: number) {
    return n.toString().padStart(2, "0");
}
